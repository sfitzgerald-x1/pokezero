"""Analyze the history-truncation probe grid against the plan's PRE-REGISTERED rules.

Reads the per-cell benchmark summary JSONs written by
``neural_cli benchmark --summary-out`` across the grid
``k ∈ {16,32,64,128} × {checkpoint...}`` and produces the verdict the probe plan
(docs/history_truncation_probe_plan.md, "Interpretation rules") pre-committed to:

  - FLAT down to some k*  (every opponent's win-rate delta vs full-history within
    2×SE, on both checkpoints)              -> deep slots decorative; adopt a
    k*-sized region (k* -> next power of two).
  - DEGRADATION at small k                  -> usage proven; keep 128.
  - MIXED / class-dependent                 -> usage-proven for the cutover
    (keep 128); record the pattern for the sibling history-compression study.

Each cell's win rate is the checkpoint's paired-both-seats head-to-head win rate
vs each opponent. The checkpoint policy id is inferred as the one policy present
in every head-to-head of a cell. The full-history baseline is k == 128 (or the
cell with no ``history_mask_k`` stamp).

This tool computes evidence; the layout decision is the owner's, in the cutover PR.

Usage:
  python scripts/analyze_history_truncation_probe.py \
      runs/probe/*.json --out runs/probe/verdict.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

FULL_K = 128


def _load_cells(paths: list[Path]) -> list[dict[str, Any]]:
    cells = []
    for path in paths:
        payload = json.loads(path.read_text())
        head_to_heads = payload.get("head_to_heads") or []
        if not head_to_heads:
            raise SystemExit(f"{path}: no head_to_heads in summary")
        # The checkpoint is the policy that plays in EVERY head-to-head.
        id_sets = [
            {h2h["first_policy_id"], h2h["second_policy_id"]} for h2h in head_to_heads
        ]
        common = set.intersection(*id_sets)
        if len(common) != 1:
            raise SystemExit(
                f"{path}: could not infer a single checkpoint policy id (candidates {common})"
            )
        checkpoint_id = next(iter(common))
        opponents: dict[str, dict[str, int]] = {}
        for h2h in head_to_heads:
            if h2h["first_policy_id"] == checkpoint_id:
                opponent = h2h["second_policy_id"]
                wins = int(h2h["first_policy_wins"])
            else:
                opponent = h2h["first_policy_id"]
                wins = int(h2h["second_policy_wins"])
            # Sum wins/games so multiple shard files for one cell aggregate cleanly.
            slot = opponents.setdefault(opponent, {"wins": 0, "games": 0})
            slot["wins"] += wins
            slot["games"] += int(h2h["games"])
        provenance = (payload.get("policy_provenance") or {}).get(checkpoint_id) or {}
        cells.append(
            {
                "path": str(path),
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": provenance.get("checkpoint_path"),
                "k": int(payload.get("history_mask_k") or FULL_K),
                "opponents": opponents,
            }
        )
    return _merge_shards(cells)


def _merge_shards(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate multiple files that share a (checkpoint, k) — i.e. shards of one cell."""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in cells:
        key = (cell["checkpoint_id"], cell["k"])
        if key not in merged:
            merged[key] = {
                "checkpoint_id": cell["checkpoint_id"],
                "checkpoint_path": cell.get("checkpoint_path"),
                "k": cell["k"],
                "opponents": {},
                "shards": 0,
            }
        target = merged[key]
        target["shards"] += 1
        if not target.get("checkpoint_path"):
            target["checkpoint_path"] = cell.get("checkpoint_path")
        for opponent, stat in cell["opponents"].items():
            slot = target["opponents"].setdefault(opponent, {"wins": 0, "games": 0})
            slot["wins"] += stat["wins"]
            slot["games"] += stat["games"]
    # Finalize win rates.
    out = []
    for cell in merged.values():
        for opponent, stat in cell["opponents"].items():
            stat["win_rate"] = stat["wins"] / stat["games"] if stat["games"] else 0.0
        out.append(cell)
    return out


def fold_foulplay(cells: list[dict[str, Any]], foulplay_root: Path) -> None:
    """Attach a synthetic ``foul-play`` opponent to each cell from foul-play probe results.

    ``foulplay_root`` contains ``<name>-k<K>/result.json`` (or partial-result.json) written by
    foundation/foulplay-k8s-probe.sh (schema pokezero.foulplay_progress.* with wins /
    completed_games / checkpoint_path). Cells are matched by (checkpoint_path, k) — the same
    checkpoint the ladder cell benchmarked.
    """
    by_key = {(c.get("checkpoint_path"), c["k"]): c for c in cells}
    matched = 0
    for result_path in sorted(foulplay_root.rglob("result.json")) + sorted(
        foulplay_root.rglob("partial-result.json")
    ):
        payload = json.loads(result_path.read_text())
        games = int(payload.get("completed_games") or 0)
        wins = int(payload.get("wins") or 0)
        checkpoint_path = payload.get("checkpoint_path")
        # k from the enclosing dir name (…-k<K>), falling back to 128.
        k = FULL_K
        for part in reversed(result_path.parts):
            if "-k" in part and part.rsplit("-k", 1)[-1].isdigit():
                k = int(part.rsplit("-k", 1)[-1])
                break
        cell = by_key.get((checkpoint_path, k))
        if cell is None or games <= 0:
            continue
        slot = cell["opponents"].setdefault("foul-play", {"wins": 0, "games": 0})
        slot["wins"] += wins
        slot["games"] += games
        slot["win_rate"] = slot["wins"] / slot["games"]
        matched += 1
    if matched:
        print(f"folded foul-play results into {matched} cells", flush=True)


def _se(win_rate: float, games: int) -> float:
    if games <= 0:
        return float("inf")
    return math.sqrt(max(win_rate * (1.0 - win_rate), 1e-9) / games)


def analyze(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_checkpoint: dict[str, dict[int, dict[str, Any]]] = {}
    for cell in cells:
        by_checkpoint.setdefault(cell["checkpoint_id"], {})[cell["k"]] = cell

    checkpoint_verdicts: dict[str, Any] = {}
    for checkpoint_id, cells_by_k in by_checkpoint.items():
        if FULL_K not in cells_by_k:
            raise SystemExit(f"{checkpoint_id}: missing full-history (k={FULL_K}) cell")
        full = cells_by_k[FULL_K]["opponents"]
        rows = []
        degraded_ks: set[int] = set()
        flat_ks: set[int] = set()
        for k in sorted(x for x in cells_by_k if x != FULL_K):
            opp_deltas = {}
            any_degraded = False
            all_flat = True
            for opponent, full_stat in full.items():
                if opponent not in cells_by_k[k]["opponents"]:
                    continue
                cur = cells_by_k[k]["opponents"][opponent]
                delta = cur["win_rate"] - full_stat["win_rate"]
                # SE of the delta between two independent win rates.
                band = 2.0 * math.sqrt(
                    _se(cur["win_rate"], cur["games"]) ** 2
                    + _se(full_stat["win_rate"], full_stat["games"]) ** 2
                )
                within = abs(delta) <= band
                degraded = delta < -band
                opp_deltas[opponent] = {
                    "win_rate": round(cur["win_rate"], 4),
                    "full_win_rate": round(full_stat["win_rate"], 4),
                    "delta": round(delta, 4),
                    "band_2se": round(band, 4),
                    "within_2se": within,
                    "degraded": degraded,
                }
                any_degraded = any_degraded or degraded
                all_flat = all_flat and within
            if any_degraded:
                degraded_ks.add(k)
            if all_flat:
                flat_ks.add(k)
            rows.append({"k": k, "all_flat": all_flat, "any_degraded": any_degraded, "opponents": opp_deltas})

        # k* = smallest k for which this k and every larger truncated k are flat.
        candidate_ks = sorted(x for x in cells_by_k if x != FULL_K)
        k_star = None
        for k in candidate_ks:
            if all(kk in flat_ks for kk in candidate_ks if kk >= k):
                k_star = k
                break
        checkpoint_verdicts[checkpoint_id] = {
            "k_star": k_star,
            "degraded_ks": sorted(degraded_ks),
            "flat_ks": sorted(flat_ks),
            "rows": rows,
        }

    # Cross-checkpoint verdict per the pre-registered asymmetry ("Degradation ... proves
    # usage"). The flat case is "flat down to SOME k*" — every checkpoint has a k below full
    # at which it (and every larger truncation) stays within 2xSE — NOT "flat to the minimum
    # tested k". A positive deviation is not degradation, so it never flips the verdict to
    # usage/mixed; only a delta < -2xSE (degraded_ks) does.
    all_have_k_star = all(v["k_star"] is not None for v in checkpoint_verdicts.values())
    any_degraded = any(v["degraded_ks"] for v in checkpoint_verdicts.values())
    all_degraded = all(v["degraded_ks"] for v in checkpoint_verdicts.values())
    k_stars = [v["k_star"] for v in checkpoint_verdicts.values() if v["k_star"] is not None]

    if all_have_k_star and not any_degraded:
        agreed_k_star = max(k_stars) if k_stars else None
        next_pow2 = 1 << (agreed_k_star - 1).bit_length() if agreed_k_star else None
        verdict = "flat"
        recommendation = (
            f"STRONG evidence deep history slots are decorative. Adopt a k*={agreed_k_star}-sized "
            f"history region (margin -> next power of two = {next_pow2}); sequence shrinks 151 -> "
            f"{23 + (next_pow2 or 0)}. Evidence-backed layout decision for the cutover."
        )
    elif all_degraded:
        verdict = "degraded"
        recommendation = (
            "Usage PROVEN — small-k degrades on both checkpoints. Keep 128 (or run one S-scale "
            "trained variant at the candidate length as the tiebreaker). Attention earns its cost."
        )
    else:
        verdict = "mixed"
        recommendation = (
            "Mixed / class-dependent. Treat as usage-proven for the cutover (keep 128); record the "
            "pattern (which opponents/checkpoints/k) as input to the sibling history-compression study."
        )

    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "k_stars_by_checkpoint": {c: v["k_star"] for c, v in checkpoint_verdicts.items()},
        "checkpoints": checkpoint_verdicts,
    }


def _print(report: dict[str, Any]) -> None:
    print(f"VERDICT: {report['verdict'].upper()}")
    print(report["recommendation"])
    for checkpoint_id, v in report["checkpoints"].items():
        print(f"\n[{checkpoint_id}]  k*={v['k_star']}  degraded_ks={v['degraded_ks']}")
        for row in v["rows"]:
            flag = "DEGRADED" if row["any_degraded"] else ("flat" if row["all_flat"] else "mixed")
            deltas = ", ".join(
                f"{opp}:{d['delta']:+.3f}{'*' if d['degraded'] else ''}"
                for opp, d in row["opponents"].items()
            )
            print(f"  k={row['k']:>3}  [{flag}]  {deltas}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+", help="Per-cell ladder benchmark summary JSON paths (globs ok).")
    parser.add_argument("--foulplay-root", type=Path, default=None,
                        help="Optional dir of foul-play probe results (<name>-k<K>/result.json) to fold in as a 'foul-play' opponent.")
    parser.add_argument("--out", type=Path, default=None, help="Optional verdict JSON output path.")
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for pattern in args.summaries:
        matched = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matched or [Path(pattern)])
    cells = _load_cells(paths)
    if args.foulplay_root is not None:
        fold_foulplay(cells, args.foulplay_root)
    report = analyze(cells)
    _print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nverdict: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
