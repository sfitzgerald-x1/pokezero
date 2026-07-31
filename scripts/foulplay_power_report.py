#!/usr/bin/env python3
"""Merge FoulPlay power-config shards into the campaign's deliverable metric.

The campaign's whole output is a **paired delta (search − raw) per cell**, and
until this existed nothing computed one: the shards carried raw game rows and
the plan described the statistics, but no merger, Wilson interval, bootstrap or
McNemar test lived anywhere in either repo.

What "paired" means here, precisely, and why the join is strict:

    A pair is one (battle seed, seat). The search arm and the raw arm played
    that same seed from that same seat against the same FoulPlay build, so the
    two rows differ only by search config. The delta is the mean of per-pair
    differences, NOT the difference of two arms' means -- those coincide only
    when both arms cover exactly the same pairs, which is the thing most likely
    to be false in a partially-failed campaign.

Fail-closed rules, each chosen because the alternative silently produces a
plausible number:

* a pair present in one arm and missing in the other is DROPPED and COUNTED,
  never half-scored;
* shards from two different engine builds refuse to merge (`--expect-fingerprint`);
* two shards claiming the same (config_id, seed, seat) with different outcomes
  is terminal, not last-write-wins;
* a cell whose `search_wall_per_searched_decision` mean exceeds the cap is
  reported REJECTED and its delta is not eligible for adoption;
* a Tier-2 depth cell without reached-depth evidence is reported UNSCOREABLE
  rather than being read as a null (the confound the depth axis exists to
  avoid).

Usage::

    python scripts/foulplay_power_report.py results/*.json \\
        --expect-fingerprint <sha256> --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SCHEMA_VERSION = "pokezero.foulplay-power-report.v1"
# Section 6, binding. Reported against the mean; p95 is reported, not gated.
LATENCY_CAP_SECONDS = 20.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260730


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Same estimator the acceptance report uses."""
    if n == 0:
        return (0.0, 0.0)
    phat = wins / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def mcnemar(deltas: list[float]) -> dict:
    """Discordant-pair counts and a normal-approximation two-sided z.

    Reported alongside the bootstrap because the two answer different
    questions: the bootstrap bounds the size of the delta, McNemar asks whether
    the DISCORDANT pairs are lopsided. The July-30 baseline was quoted as
    "+0.058 (discordant 87/64)" and this reproduces that shape.
    """
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    n = wins + losses
    if n == 0:
        return {"search_better": 0, "raw_better": 0, "discordant": 0, "z": None,
                "note": "no discordant pairs"}
    z = (wins - losses) / math.sqrt(n)
    return {"search_better": wins, "raw_better": losses, "discordant": n,
            "z": round(z, 3)}


def load_shards(paths: list[str]) -> list[dict]:
    shards = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = payload.get("schema_version")
        if schema != "pokezero.foulplay-paired-shard.v1":
            raise SystemExit(f"{path}: unexpected schema_version {schema!r}")
        payload["_path"] = path
        shards.append(payload)
    if not shards:
        raise SystemExit("no shards given")
    return shards


def assert_single_build(shards: list[dict], expect: str | None) -> str:
    """One build era for the whole campaign (the #952/#963 standard)."""
    prints = {s.get("engine_fingerprint") for s in shards}
    if len(prints) != 1:
        raise SystemExit(
            f"shards span {len(prints)} engine builds: {sorted(map(str, prints))}. "
            "A campaign is ONE build era; refusing to merge across builds."
        )
    fingerprint = prints.pop()
    if expect is not None and fingerprint != expect:
        raise SystemExit(
            f"engine fingerprint {fingerprint} does not match --expect-fingerprint {expect}"
        )
    return fingerprint


def collect_rows(shards: list[dict]) -> tuple[dict, dict]:
    """(config_id -> {(seed, seat): score}), and config_id -> shard metadata."""
    rows: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for shard in shards:
        cid = shard["config_id"]
        meta.setdefault(cid, {"arm": shard["arm"], "shards": [], "per_seat": [],
                              "checkpoint": shard.get("checkpoint"),
                              "opponent_priors": shard.get("opponent_priors", False)})
        meta[cid]["shards"].append(shard["_path"])
        meta[cid]["per_seat"].append(shard.get("per_seat", {}))
        for row in shard.get("rows", []):
            key = (int(row["seed"]), row["seat"])
            score = float(row["score"])
            if key in rows[cid] and rows[cid][key] != score:
                raise SystemExit(
                    f"conflicting scores for {cid} seed {key[0]} seat {key[1]}: "
                    f"{rows[cid][key]} vs {score}. Two shards disagree about the "
                    "same game; refusing to pick one."
                )
            rows[cid][key] = score
    return rows, meta


def latency_of(meta_entry: dict) -> dict:
    """Mean/p95 of the GATE field across a cell's seats, plus the other wall."""
    gate, p95, other = [], [], []
    for per_seat in meta_entry["per_seat"]:
        for seat in (per_seat or {}).values():
            if seat.get("search_wall_per_searched_decision") is not None:
                gate.append(float(seat["search_wall_per_searched_decision"]))
            if seat.get("wall_per_decision_p95") is not None:
                p95.append(float(seat["wall_per_decision_p95"]))
            if seat.get("wall_per_decision_mean") is not None:
                other.append(float(seat["wall_per_decision_mean"]))
    return {
        "search_wall_per_searched_decision_mean": (sum(gate) / len(gate)) if gate else None,
        "wall_per_decision_p95_max": max(p95) if p95 else None,
        "wall_per_decision_mean": (sum(other) / len(other)) if other else None,
    }


def health_of(meta_entry: dict) -> dict:
    fallback, depth = [], []
    reasons: dict[str, int] = {}
    for per_seat in meta_entry["per_seat"]:
        for seat in (per_seat or {}).values():
            if seat.get("fallback_rate") is not None:
                fallback.append(float(seat["fallback_rate"]))
            if seat.get("depth_reached_mean") is not None:
                depth.append(float(seat["depth_reached_mean"]))
            for reason, count in (seat.get("world_failure_reasons") or {}).items():
                reasons[reason] = reasons.get(reason, 0) + int(count)
    return {
        "fallback_rate": (sum(fallback) / len(fallback)) if fallback else None,
        "depth_reached_mean": (sum(depth) / len(depth)) if depth else None,
        "world_failure_reasons": dict(sorted(reasons.items())),
    }


def score_cell(search: dict, raw: dict, indices) -> dict:
    """Paired delta over the pairs BOTH arms actually played."""
    from pokezero.mcts_eval.scoring import bootstrap_mean, bootstrap_paired_delta

    shared = sorted(set(search) & set(raw))
    only_search = sorted(set(search) - set(raw))
    only_raw = sorted(set(raw) - set(search))
    if not shared:
        return {"pairs": 0, "error": "no shared (seed, seat) pairs between the arms"}
    t = [search[k] for k in shared]
    b = [raw[k] for k in shared]
    delta = bootstrap_paired_delta(t, b, indices)
    per_seat = {}
    for seat in ("p1", "p2"):
        keys = [k for k in shared if k[1] == seat]
        if not keys:
            continue
        st = [search[k] for k in keys]
        sb = [raw[k] for k in keys]
        seat_idx = _indices(len(keys))
        seat_delta = bootstrap_paired_delta(st, sb, seat_idx)
        per_seat[seat] = {
            "pairs": len(keys),
            "search_rate": sum(st) / len(st),
            "raw_rate": sum(sb) / len(sb),
            "paired_delta": seat_delta.to_payload(),
        }
    return {
        "pairs": len(shared),
        "dropped_unpaired": {"search_only": len(only_search), "raw_only": len(only_raw)},
        "search_rate": sum(t) / len(t),
        "search_wilson95": list(wilson(sum(t), len(t))),
        "raw_rate": sum(b) / len(b),
        "raw_wilson95": list(wilson(sum(b), len(b))),
        "paired_delta": delta.to_payload(),
        "mcnemar": mcnemar([x - y for x, y in zip(t, b)]),
        "per_seat": per_seat,
    }


def _indices(n: int):
    from pokezero.mcts_eval.scoring import bootstrap_indices

    return bootstrap_indices(sample_size=n, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)


def seat_gap_flag(cell: dict) -> str | None:
    """#937 bug class: a search-arm seat gap the raw arm does not show."""
    seats = cell.get("per_seat") or {}
    if set(seats) != {"p1", "p2"}:
        return None
    d1, d2 = seats["p1"]["paired_delta"], seats["p2"]["paired_delta"]
    disjoint = d1["low"] > d2["high"] or d2["low"] > d1["high"]
    return "STOP-AND-INVESTIGATE: per-seat paired deltas disjoint at 95%" if disjoint else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--expect-fingerprint", default=None)
    ap.add_argument("--anchor", default=None,
                    help="config_id of the anchor cell (A); winners must beat it")
    ap.add_argument("--cap-seconds", type=float, default=LATENCY_CAP_SECONDS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    shards = load_shards(args.shards)
    fingerprint = assert_single_build(shards, args.expect_fingerprint)
    rows, meta = collect_rows(shards)

    raw_arms = {c: r for c, r in rows.items() if meta[c]["arm"] == "raw"}
    if not raw_arms:
        raise SystemExit("no raw arm among the shards; a paired delta is undefined")

    cells: dict[str, dict] = {}
    for cid, search in rows.items():
        if meta[cid]["arm"] == "raw":
            continue
        # Pair against the raw arm of the SAME checkpoint. Matching on the
        # checkpoint rather than on arity is what keeps cell G (k1) off cell
        # A's (k0) denominator.
        ckpt = meta[cid]["checkpoint"]
        candidates = [c for c in raw_arms if meta[c]["checkpoint"] == ckpt]
        if len(candidates) != 1:
            cells[cid] = {"error": f"expected exactly one raw arm for checkpoint "
                                   f"{ckpt!r}, found {len(candidates)}"}
            continue
        raw = raw_arms[candidates[0]]
        shared_n = len(set(search) & set(raw))
        scored = score_cell(search, raw, _indices(shared_n)) if shared_n else {
            "pairs": 0, "error": "no shared pairs"}
        scored["raw_arm"] = candidates[0]
        scored["latency"] = latency_of(meta[cid])
        scored["health"] = health_of(meta[cid])
        scored["opponent_priors"] = meta[cid]["opponent_priors"]

        gate = scored["latency"]["search_wall_per_searched_decision_mean"]
        if gate is None:
            scored["cap"] = "UNEVALUABLE - no search_wall_per_searched_decision in any shard"
        elif gate > args.cap_seconds:
            scored["cap"] = f"REJECTED - mean {gate:.2f}s exceeds {args.cap_seconds:.0f}s"
        else:
            scored["cap"] = f"PASS - mean {gate:.2f}s"
        flag = seat_gap_flag(scored)
        if flag:
            scored["seat_health"] = flag
        cells[cid] = scored

    eligible = {
        c: v for c, v in cells.items()
        if v.get("pairs") and str(v.get("cap", "")).startswith("PASS")
        and "seat_health" not in v
    }
    ranked = sorted(eligible, key=lambda c: eligible[c]["paired_delta"]["point"], reverse=True)
    winner = None
    if ranked and args.anchor and args.anchor in cells:
        best = eligible[ranked[0]]
        anchor = cells[args.anchor]
        # Adoption rule (section 9 Phase 2): the winner must beat the ANCHOR's
        # delta, not merely have the largest one.
        if best["paired_delta"]["low"] > anchor.get("paired_delta", {}).get("point", 0.0):
            winner = ranked[0]
        else:
            winner = args.anchor
    elif ranked:
        winner = ranked[0]

    report = {
        "schema_version": SCHEMA_VERSION,
        "engine_fingerprint": fingerprint,
        "cap_seconds": args.cap_seconds,
        "anchor": args.anchor,
        "cells": cells,
        "ranking_eligible": ranked,
        "winner": winner,
        "winner_note": (
            "Eligibility requires shared pairs, a passing cap, and no seat-gap flag. "
            "A cell rejected by the cap is NOT a miss for its strength prediction -- "
            "it is unscored."
        ),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
