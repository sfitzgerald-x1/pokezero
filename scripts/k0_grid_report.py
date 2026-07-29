"""Aggregate the k0 depth-grid shards into cells, with Wilson intervals and slopes.

Shards of a cell share a seed BLOCK, never a seed, so pooling them is just
summing Bernoulli trials. Every shard carries its own provenance (which encoder
tables it ran against, and whether those tables agreed with the checkpoint), and
pooling refuses to mix shards whose provenance differs -- that is the whole
point of the campaign, so silently averaging across it would destroy the result.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Draws count as half a success, so `successes` is a float."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_cells(results_dir: Path, prefix: str):
    cells = defaultdict(lambda: {
        "wins": 0.0, "n": 0, "decisions": 0, "fallbacks": 0, "wall": 0.0,
        "shards": [], "seeds": set(), "provenance": None, "drift": None,
        # Seat is reported separately because the crate carried a seat-CONSTANT
        # model-value inversion (parity lane, PR #937): on p2-seated roots the
        # model's leaf values entered the side-one-absolute tree unreflected. On an
        # unfixed build the p1 half of every cell measures fixed-build behaviour and
        # the p2 half characterises the bug, so pooling them averages two different
        # experiments. Seat is a function of the seed, so the split is exactly 50/50
        # and the two halves are still paired across cells.
        "seat_wins": defaultdict(float), "seat_n": defaultdict(int),
    })
    for path in sorted(results_dir.glob(f"{prefix}*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = payload.get("arm") or "?"
        depth = payload.get("depth")
        key = (arm, depth)
        cell = cells[key]
        prov = payload.get("provenance") or {}
        tables = prov.get("tables_path")
        drift = prov.get("mask_drift")
        if cell["provenance"] is None:
            cell["provenance"] = tables
            cell["drift"] = drift
        elif cell["provenance"] != tables:
            raise SystemExit(
                f"{path.name}: shard provenance disagrees within cell {key}: "
                f"{tables} vs {cell['provenance']}"
            )
        cell["wins"] += payload["search_wins"] + 0.5 * payload["draws"]
        cell["n"] += payload["games"]
        cell["decisions"] += payload["total_decisions"]
        cell["fallbacks"] += payload["fallback_decisions"]
        cell["wall"] += payload["search_wall_per_decision"] * payload["total_decisions"]
        cell["shards"].append(path.name)
        for game in payload["per_game"]:
            seed = game["seed"]
            if seed in cell["seeds"]:
                raise SystemExit(f"{path.name}: seed {seed} double-counted in cell {key}")
            cell["seeds"].add(seed)
            seat = game["search_seat"]
            cell["seat_n"][seat] += 1
            if game["winner"] == seat:
                cell["seat_wins"][seat] += 1.0
            elif game["winner"] is None:
                cell["seat_wins"][seat] += 0.5
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--contrast", default=None,
                    help="ARM:ARM cross-model deep-cell contrast, e.g. c:k64c")
    args = ap.parse_args()

    cells = load_cells(Path(args.results), args.prefix)
    rows = []
    for (arm, depth), cell in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        n = cell["n"]
        score = cell["wins"] / n if n else 0.0
        low, high = wilson(cell["wins"], n)
        row = {
            "arm": arm, "depth": depth, "n": n,
            "score": round(score, 4),
            "wilson_low": round(low, 3), "wilson_high": round(high, 3),
            "half_width": round((high - low) / 2, 3),
            "fallback_rate": round(cell["fallbacks"] / max(1, cell["decisions"]), 5),
            "s_per_decision": round(cell["wall"] / max(1, cell["decisions"]), 3),
            "shards": len(cell["shards"]),
            "tables": (cell["provenance"] or "").split("/tables/")[-1] or "(control)",
            "mask_drift": sorted((cell["drift"] or {}).keys()) or [],
        }
        for seat in ("p1", "p2"):
            sn = cell["seat_n"][seat]
            sw = cell["seat_wins"][seat]
            slo, shi = wilson(sw, sn)
            row[f"{seat}_n"] = sn
            row[f"{seat}_score"] = round(sw / sn, 4) if sn else None
            row[f"{seat}_low"] = round(slo, 3)
            row[f"{seat}_high"] = round(shi, 3)
        rows.append(row)

    width = max(len(f"{r['arm']}-d{r['depth']}") for r in rows) if rows else 8
    print(f"{'cell'.ljust(width)}  {'n':>4}  {'score':>6}  {'Wilson 95%':>16}  "
          f"{'fb':>7}  {'s/dec':>7}  tables")
    for r in rows:
        cell_id = f"{r['arm']}-d{r['depth']}"
        ci = f"[{r['wilson_low']:.3f}, {r['wilson_high']:.3f}]"
        drift = "  DRIFT:" + ",".join(r["mask_drift"]) if r["mask_drift"] else ""
        print(f"{cell_id.ljust(width)}  {r['n']:>4}  {r['score']:>6.3f}  {ci:>16}  "
              f"{r['fallback_rate']:>7.4f}  {r['s_per_decision']:>7.2f}  {r['tables']}{drift}")

    print("\n=== per-seat split (p1 = fixed-build behaviour; p2 = carries the inversion) ===")
    width2 = max(len(f"{r['arm']}-d{r['depth']}") for r in rows) if rows else 8
    print(f"{'cell'.ljust(width2)}  {'p1 score':>9}  {'p1 Wilson':>16}  "
          f"{'p2 score':>9}  {'p2 Wilson':>16}  {'p1-p2':>7}")
    for r in rows:
        cell_id = f"{r['arm']}-d{r['depth']}"
        if r["p1_score"] is None or r["p2_score"] is None:
            continue
        gap = r["p1_score"] - r["p2_score"]
        ci1 = f"[{r['p1_low']:.3f}, {r['p1_high']:.3f}]"
        ci2 = f"[{r['p2_low']:.3f}, {r['p2_high']:.3f}]"
        print(f"{cell_id.ljust(width2)}  {r['p1_score']:>9.3f}  {ci1:>16}  "
              f"{r['p2_score']:>9.3f}  {ci2:>16}  {gap:>+7.3f}")

    def _slope(label, arm, depths, score_key, hw_fn):
        lo, hi = min(depths), max(depths)
        a, b = depths[lo], depths[hi]
        if a[score_key] is None or b[score_key] is None:
            return
        delta = b[score_key] - a[score_key]
        # Independent cells (disjoint games), so the difference's SE is the
        # quadrature sum. Half-widths are Wilson, which is fine at these n.
        se = math.sqrt((hw_fn(a) / 1.96) ** 2 + (hw_fn(b) / 1.96) ** 2)
        z = delta / se if se else 0.0
        print(f"  {label:12s} d{lo}={a[score_key]:.3f} -> d{hi}={b[score_key]:.3f}  "
              f"delta={delta:+.3f}  z={z:+.2f}  "
              f"{'DECAY' if z < -1.96 else 'FLAT (no significant decay)'}")

    by_arm = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["depth"]] = r

    print("\n=== depth slope per arm (d1 -> deepest), POOLED ===")
    for arm, depths in sorted(by_arm.items()):
        if len(depths) >= 2:
            _slope(arm, arm, depths, "score", lambda r: r["half_width"])

    # The decisive read: a seat-CONSTANT inversion can only depress the p2 half.
    # If the decay is present on p1 alone it cannot be the inversion.
    for seat in ("p1", "p2"):
        print(f"\n=== depth slope per arm, {seat} SEAT ONLY ===")
        for arm, depths in sorted(by_arm.items()):
            if len(depths) >= 2:
                _slope(f"{arm}/{seat}", arm, depths, f"{seat}_score",
                       lambda r, s=seat: (r[f"{s}_high"] - r[f"{s}_low"]) / 2)

    # The experiment's actual question: does a HISTORY-FREE checkpoint decay with
    # depth the way a history-carrying one does? Each model is scored against its
    # own raw policy, so 0.500 is the null for both and the comparison is not
    # confounded by the two checkpoints differing in strength.
    #
    # Single cells at n=100 cannot resolve a 0.10-0.12 difference (half-width
    # ~0.10), so the deep cells are pooled. They are disjoint games, so this is
    # just a larger binomial, and pooling d4+d6 is the natural "deep" contrast
    # rather than a threshold chosen after seeing the numbers.
    if args.contrast:
        left, right = args.contrast.split(":")
        print(f"\n=== cross-model contrast: {left} vs {right}, deep cells (d4+d6) ===")
        for seat_label, score_key, n_key in (
            ("pooled", "score", "n"), ("p1 only", "p1_score", "p1_n"),
            ("p2 only", "p2_score", "p2_n"),
        ):
            agg = {}
            for arm in (left, right):
                wins = n = 0.0
                for depth in (4, 6):
                    r = by_arm.get(arm, {}).get(depth)
                    if r and r[score_key] is not None:
                        wins += r[score_key] * r[n_key]
                        n += r[n_key]
                agg[arm] = (wins, n)
            (wl, nl), (wr, nr) = agg[left], agg[right]
            if not nl or not nr:
                continue
            pl, pr = wl / nl, wr / nr
            se = math.sqrt(pl * (1 - pl) / nl + pr * (1 - pr) / nr)
            z = (pl - pr) / se if se else 0.0
            verdict = "DIFFERENT" if abs(z) > 1.96 else "not separable at 95%"
            print(f"  {seat_label:8s} {left}={pl:.3f} (n={int(nl)})  {right}={pr:.3f} "
                  f"(n={int(nr)})  delta={pl - pr:+.3f}  z={z:+.2f}  {verdict}")

        # The level contrast above is NOT baseline-free. Raw-vs-raw is 0.500 only in
        # expectation: with two identical deterministic policies the winner of a
        # seeded game is a property of the game, so each checkpoint has its own
        # measured null (k0 0.570, k64 0.485 on this seed block). A level difference
        # between two checkpoints inherits the difference between their nulls.
        #
        # The SLOPE does not: a per-checkpoint offset is constant in depth and
        # cancels in d1 -> deep. So the hypothesis "history-carrying models decay
        # with depth and Markov ones do not" is a claim about slopes, and this is
        # the test of it.
        print(f"\n=== slope difference (baseline-free): {left} vs {right} ===")
        for seat_label, score_key, n_key in (
            ("pooled", "score", "n"), ("p1 only", "p1_score", "p1_n"),
            ("p2 only", "p2_score", "p2_n"),
        ):
            slopes = {}
            for arm in (left, right):
                shallow = by_arm.get(arm, {}).get(1)
                if not shallow or shallow[score_key] is None:
                    continue
                dw = dn = 0.0
                for depth in (4, 6):
                    r = by_arm.get(arm, {}).get(depth)
                    if r and r[score_key] is not None:
                        dw += r[score_key] * r[n_key]
                        dn += r[n_key]
                if not dn:
                    continue
                p1_, n1_ = shallow[score_key], shallow[n_key]
                p2_, n2_ = dw / dn, dn
                slopes[arm] = (
                    p2_ - p1_,
                    p1_ * (1 - p1_) / n1_ + p2_ * (1 - p2_) / n2_,
                )
            if len(slopes) < 2:
                continue
            (sl, vl), (sr, vr) = slopes[left], slopes[right]
            se = math.sqrt(vl + vr)
            z = (sl - sr) / se if se else 0.0
            verdict = "DIFFERENT" if abs(z) > 1.96 else "suggestive, NOT significant"
            print(f"  {seat_label:8s} slope({left})={sl:+.3f}  slope({right})={sr:+.3f}  "
                  f"diff={sl - sr:+.3f}  z={z:+.2f}  {verdict}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
