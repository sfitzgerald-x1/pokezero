#!/usr/bin/env python3
"""Merge §8 acceptance shards and report the verdict, split by seat.

Reads the shard JSONs written by ``scripts/mcts_acceptance_h2h.py`` and reports,
for every arm:

* **per seat** — score and Wilson 95% interval (this is what §11 showed the
  pooled number hides);
* **pooled** — the mirrored-PAIR mean with a deterministic percentile bootstrap,
  via ``pokezero.mcts_eval.scoring`` (fail-closed: a pair missing a seat is an
  error, never a silently half-scored row);
* the acceptance verdict against the §8 bar, ``> 0.500`` for the search arm.

The prediction this run was staged to falsify (docs/mcts_degradation_findings.md
§11.8): **p1 stays where it is and p2 rises to meet it.** If p2 does not
recover, §10 and §11 both retract.

Usage::

    python scripts/mcts_acceptance_report.py shards/*.json
    python scripts/mcts_acceptance_report.py --json report.json shards/*.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.mcts_eval.scoring import (  # noqa: E402
    GameResult,
    MergeError,
    bootstrap_indices,
    bootstrap_mean,
    merge_game_results,
    outcome_record,
    pair_scores,
    parity_label,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260729


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load(paths: list[str]) -> tuple[dict[str, list[GameResult]], dict[str, set[str]]]:
    by_arm: dict[str, list[GameResult]] = defaultdict(list)
    fingerprints: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "pokezero.mcts-acceptance-shard.v1":
            raise SystemExit(f"{path}: not an acceptance shard")
        arm = payload["arm"]
        fingerprints[arm].add(payload["engine_fingerprint"])
        for row in payload["results"]:
            by_arm[arm].append(
                GameResult(
                    config_id=row["config_id"],
                    seed=int(row["seed"]),
                    seat=row["seat"],
                    outcome=row["outcome"],
                    turns=int(row["turns"]),
                    provenance_sha256=row["provenance_sha256"],
                    opponent_crashed=bool(row.get("opponent_crashed")),
                )
            )
    return by_arm, fingerprints


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--expect-fingerprint", default=None,
                    help="refuse to report if any shard was produced by a different build")
    args = ap.parse_args(argv)

    by_arm, fingerprints = load(args.shards)
    summary: dict[str, dict] = {}

    for arm in sorted(by_arm):
        seen = fingerprints[arm]
        if len(seen) != 1:
            raise SystemExit(
                f"arm {arm!r} mixes {len(seen)} engine builds: {sorted(seen)}. "
                "A merged number across builds is not a measurement."
            )
        fingerprint = next(iter(seen))
        if args.expect_fingerprint and fingerprint != args.expect_fingerprint:
            raise SystemExit(
                f"arm {arm!r} was produced by engine {fingerprint[:16]}, staged config "
                f"expects {args.expect_fingerprint[:16]}"
            )
        results = by_arm[arm]
        config_id = results[0].config_id
        merged = merge_game_results(results)

        # Complete pairs only; an incomplete pair is reported, never scored.
        by_seed: dict[int, set[str]] = defaultdict(set)
        for (_cfg, seed, seat) in merged:
            by_seed[seed].add(seat)
        complete = sorted(s for s, seats in by_seed.items() if seats == {"p1", "p2"})
        incomplete = sorted(s for s, seats in by_seed.items() if seats != {"p1", "p2"})

        try:
            pairs = pair_scores(results, seeds=complete, config_id=config_id)
        except MergeError as error:
            raise SystemExit(f"arm {arm!r}: {error}") from error

        indices = bootstrap_indices(
            sample_size=len(pairs), resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
        )
        pooled = bootstrap_mean(pairs, indices)

        seats: dict[str, dict] = {}
        for seat in ("p1", "p2"):
            values = [merged[(config_id, seed, seat)].score for seed in complete]
            total, n = sum(values), len(values)
            low, high = wilson(total, n)
            seats[seat] = {
                "n": n,
                "score": total / n if n else float("nan"),
                "wilson95": [low, high],
            }

        summary[arm] = {
            "config_id": config_id,
            "engine_fingerprint": fingerprint,
            "complete_pairs": len(complete),
            "incomplete_pairs": incomplete,
            "per_seat": seats,
            "pooled_pair_mean": pooled.to_payload(),
            "parity_label": parity_label(pooled),
            "outcomes": outcome_record(results, config_id=config_id),
        }

        print(f"\n=== arm: {arm}  ({config_id}) ===")
        print(f"engine fingerprint : {fingerprint[:16]}")
        print(f"complete pairs     : {len(complete)}"
              + (f"   INCOMPLETE: {incomplete}" if incomplete else ""))
        for seat in ("p1", "p2"):
            entry = seats[seat]
            low, high = entry["wilson95"]
            print(f"  {seat} seat  n={entry['n']:>4}  score={entry['score']:.3f}  "
                  f"Wilson95 [{low:.3f}, {high:.3f}]")
        print(f"  pooled pair mean  {pooled.point:.3f}  "
              f"bootstrap95 [{pooled.low:.3f}, {pooled.high:.3f}]  -> {parity_label(pooled)}")
        if arm == "search":
            bar = pooled.point > 0.500
            print(f"  §8 acceptance bar (> 0.500): {'MET' if bar else 'NOT MET'} "
                  "(point estimate; read the interval and the seat split before claiming it)")

    if "search" in summary and "control" in summary:
        s, c = summary["search"], summary["control"]
        print("\n=== prediction under test (§11.8) ===")
        print("  p1 stays put, p2 rises to meet it; p2 non-recovery => §10 and §11 retract.")
        for seat in ("p1", "p2"):
            print(f"  {seat}: control {c['per_seat'][seat]['score']:.3f}   "
                  f"search {s['per_seat'][seat]['score']:.3f}   "
                  f"delta {s['per_seat'][seat]['score'] - c['per_seat'][seat]['score']:+.3f}")
        gap = s["per_seat"]["p1"]["score"] - s["per_seat"]["p2"]["score"]
        print(f"  search seat gap (p1 - p2): {gap:+.3f}   "
              f"(pre-fix d4-s1024 grid cells ran +0.34 to +0.54)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
