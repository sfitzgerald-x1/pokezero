#!/usr/bin/env python3
"""Score the named raw PokeZero vs FoulPlay budget calibration.

The contention instrument bounds an opponent-throughput shortfall; this is the
separate closing experiment that converts its *specified* 24% FoulPlay
per-battle budget cut into score units. It is intentionally not the generic
power report:

* both conditions run the same raw PokeZero checkpoint, mirrored over identical
  seeds; the p1/p2 seat readings for each seed are averaged into one
  independent paired observation;
* the only treatment is FoulPlay's budget, exactly 1000 ms versus 760 ms;
* every shard carries a named calibration and resource-layout identity, so a
  different opponent budget, raw checkpoint, or host allocation cannot pool;
* incomplete, duplicate, or non-mirrored rows are terminal. They are not
  dropped, because a missing side of a calibration pair can look like a score
  change.

Positive ``reduced_minus_baseline`` means that giving FoulPlay 760 ms raised
the raw PokeZero score, i.e. it quantifies the expected weakening direction.
The report is an estimate, not a claim that the original throughput bound has
already been translated into a universal win-rate bound.
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from foulplay_paired_eval import (  # noqa: E402
    FOULPLAY_BUDGET_CALIBRATION_BASELINE_MS,
    FOULPLAY_BUDGET_CALIBRATION_CUT_FRACTION,
    FOULPLAY_BUDGET_CALIBRATION_REDUCED_MS,
    FOULPLAY_BUDGET_CALIBRATION_SCHEMA_VERSION,
    SCHEMA_VERSION as PAIRED_SHARD_SCHEMA_VERSION,
    SEATS,
    THREAD_PIN_ENV,
    foulplay_budget_calibration_config_id,
)
from pokezero.foulplay_bridge import (  # noqa: E402
    FOULPLAY_THINK_SCHEMA_VERSION,
    foulplay_think_reading_status,
)
from pokezero.engine_search import require_rollout_leaf_document_schema  # noqa: E402

SCHEMA_VERSION = "pokezero.foulplay-budget-calibration-report.v1"
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20260818
MIN_PAIRS = 400


def _exact_int(value: object, *, path: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{path}: {field} must be an integer, got {value!r}")
    return value


def _finite_score(value: object, *, path: str, key: tuple[int, str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{path}: row {key} has no numeric score")
    score = float(value)
    if not math.isfinite(score):
        raise SystemExit(f"{path}: row {key} has non-finite score {value!r}")
    return score


def load_shards(paths: list[str]) -> list[dict]:
    shards: list[dict] = []
    for name in paths:
        path = Path(name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{path}: cannot read JSON shard: {exc}") from exc
        if payload.get("schema_version") != PAIRED_SHARD_SCHEMA_VERSION:
            raise SystemExit(
                f"{path}: expected paired-shard schema {PAIRED_SHARD_SCHEMA_VERSION!r}, "
                f"got {payload.get('schema_version')!r}"
            )
        # The calibration pools rows, so it is a reader for the rollout-leaf document
        # shape too. A malformed inherited block must not become a new banked artifact
        # merely because this experiment itself runs raw.
        require_rollout_leaf_document_schema(payload)
        payload["_path"] = str(path)
        shards.append(payload)
    if not shards:
        raise SystemExit("no shards supplied")
    return shards


def _calibration_metadata(shard: dict) -> dict:
    path = shard["_path"]
    if shard.get("arm") != "raw":
        raise SystemExit(f"{path}: calibration accepts only raw PokeZero shards")
    if shard.get("opponent_policy_mode", "foul-play") != "foul-play":
        raise SystemExit(f"{path}: calibration requires FoulPlay as the opponent")
    metadata = shard.get("foulplay_budget_calibration")
    if not isinstance(metadata, dict):
        raise SystemExit(f"{path}: missing foulplay_budget_calibration metadata")
    if metadata.get("schema_version") != FOULPLAY_BUDGET_CALIBRATION_SCHEMA_VERSION:
        raise SystemExit(f"{path}: unsupported calibration metadata schema")
    required = (
        "calibration_id",
        "resource_layout",
        "checkpoint_tag",
        "condition",
        "foulplay_search_time_ms",
        "baseline_foulplay_search_time_ms",
        "reduced_foulplay_search_time_ms",
        "budget_cut_fraction",
        "thread_pin",
        "config_id",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise SystemExit(f"{path}: calibration metadata missing {missing}")
    if (
        metadata["baseline_foulplay_search_time_ms"]
        != FOULPLAY_BUDGET_CALIBRATION_BASELINE_MS
        or metadata["reduced_foulplay_search_time_ms"]
        != FOULPLAY_BUDGET_CALIBRATION_REDUCED_MS
        or metadata["budget_cut_fraction"] != FOULPLAY_BUDGET_CALIBRATION_CUT_FRACTION
    ):
        raise SystemExit(
            f"{path}: calibration must be the named 1000-ms -> 760-ms (24%) cut"
        )
    condition = metadata["condition"]
    budget = _exact_int(
        metadata["foulplay_search_time_ms"], path=path, field="metadata budget"
    )
    expected_budget = {
        "baseline": FOULPLAY_BUDGET_CALIBRATION_BASELINE_MS,
        "reduced": FOULPLAY_BUDGET_CALIBRATION_REDUCED_MS,
    }.get(condition)
    if expected_budget is None or budget != expected_budget:
        raise SystemExit(
            f"{path}: condition {condition!r} does not match its named budget {budget!r}"
        )
    if shard.get("foulplay_search_time_ms") != budget:
        raise SystemExit(
            f"{path}: shard FoulPlay budget does not match calibration metadata"
        )
    expected_id = foulplay_budget_calibration_config_id(
        checkpoint=str(shard.get("checkpoint")),
        checkpoint_tag_value=str(metadata["checkpoint_tag"]),
        calibration_id=str(metadata["calibration_id"]),
        layout=str(metadata["resource_layout"]),
        foulplay_search_time_ms=budget,
    )
    if metadata["config_id"] != expected_id or shard.get("config_id") != expected_id:
        raise SystemExit(
            f"{path}: calibration config_id does not bind its checkpoint, budget, and layout"
        )
    if metadata["thread_pin"] != dict(sorted(THREAD_PIN_ENV.items())):
        raise SystemExit(f"{path}: calibration thread pin does not match the driver contract")
    return metadata


def _require_uniform_run_identity(shards: list[dict], metadata: list[dict]) -> None:
    """One raw policy/build/layout run; only the FoulPlay budget may differ."""

    fields = (
        ("calibration_id", lambda s, m: m["calibration_id"]),
        ("resource_layout", lambda s, m: m["resource_layout"]),
        ("checkpoint_tag", lambda s, m: m["checkpoint_tag"]),
        ("checkpoint", lambda s, m: s.get("checkpoint")),
        ("checkpoint_sha256", lambda s, m: s.get("checkpoint_sha256")),
        ("engine_fingerprint", lambda s, m: s.get("engine_fingerprint")),
        ("commit", lambda s, m: s.get("commit")),
        ("thread_pin", lambda s, m: m["thread_pin"]),
    )
    for label, reader in fields:
        values = {
            json.dumps(reader(shard, meta), sort_keys=True)
            for shard, meta in zip(shards, metadata, strict=True)
        }
        if len(values) != 1 or next(iter(values)) in {'null', '""'}:
            raise SystemExit(
                f"calibration shards do not carry one non-empty {label}; refusing to "
                "attribute a budget effect across different raw-policy executions"
            )


def _condition_rows(
    shards: list[dict], metadata: list[dict]
) -> tuple[dict[str, dict[tuple[int, str], float]], dict[str, int]]:
    rows: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
    shard_counts: dict[str, int] = defaultdict(int)
    for shard, meta in zip(shards, metadata, strict=True):
        path = shard["_path"]
        condition = str(meta["condition"])
        shard_counts[condition] += 1
        start = _exact_int(shard.get("seed_start"), path=path, field="seed_start")
        pairs = _exact_int(shard.get("pairs"), path=path, field="pairs")
        if pairs <= 0:
            raise SystemExit(f"{path}: pairs must be positive")
        expected = {
            (seed, seat)
            for seed in range(start, start + pairs)
            for seat in SEATS
        }
        payload_rows = shard.get("rows")
        if not isinstance(payload_rows, list):
            raise SystemExit(f"{path}: rows must be a list")
        observed: dict[tuple[int, str], float] = {}
        for row in payload_rows:
            if not isinstance(row, dict):
                raise SystemExit(f"{path}: row is not an object")
            seed = _exact_int(row.get("seed"), path=path, field="row seed")
            seat = row.get("seat")
            if seat not in SEATS:
                raise SystemExit(f"{path}: row {(seed, seat)!r} has invalid seat")
            key = (seed, seat)
            if key in observed:
                raise SystemExit(f"{path}: duplicate row {key}")
            observed[key] = _finite_score(row.get("score"), path=path, key=key)
        if set(observed) != expected:
            raise SystemExit(
                f"{path}: rows are not the exact mirrored seed band; missing "
                f"{sorted(expected - set(observed))[:3]}, extra "
                f"{sorted(set(observed) - expected)[:3]}"
            )
        seats = shard.get("per_seat")
        if not isinstance(seats, dict) or set(seats) != set(SEATS):
            raise SystemExit(f"{path}: per_seat must contain exactly p1 and p2")
        for seat in SEATS:
            block = seats[seat]
            if not isinstance(block, dict) or not isinstance(block.get("foulplay_think"), dict):
                raise SystemExit(f"{path}: {seat} lacks the FoulPlay think witness")
            header = block["foulplay_think"]
            if header.get("schema_version") != FOULPLAY_THINK_SCHEMA_VERSION:
                raise SystemExit(f"{path}: {seat} has an unsupported FoulPlay think schema")
            expected_budget = {
                "baseline": FOULPLAY_BUDGET_CALIBRATION_BASELINE_MS,
                "reduced": FOULPLAY_BUDGET_CALIBRATION_REDUCED_MS,
            }[condition]
            configured_budget = _exact_int(
                header.get("budget_ms_configured"),
                path=path,
                field=f"{seat} FoulPlay think configured budget",
            )
            if configured_budget != expected_budget:
                raise SystemExit(
                    f"{path}: {seat} FoulPlay think configured budget "
                    f"{configured_budget} does not match the {condition} condition"
                )
            reading = block.get("foulplay_think_reading")
            expected_reading = foulplay_think_reading_status(header)
            if reading != expected_reading or not expected_reading["usable"]:
                raise SystemExit(
                    f"{path}: {seat} lacks an admissible FoulPlay think reading"
                )
        for key, score in observed.items():
            if key in rows[condition]:
                raise SystemExit(
                    f"{path}: duplicate calibration pair {key} in {condition}; "
                    "a repeated shard must not amplify one condition"
                )
            rows[condition][key] = score
    if set(rows) != {"baseline", "reduced"}:
        raise SystemExit(
            f"calibration needs exactly baseline and reduced conditions, found {sorted(rows)}"
        )
    if set(rows["baseline"]) != set(rows["reduced"]):
        raise SystemExit(
            "baseline and reduced conditions do not cover the exact same (seed, seat) pairs"
        )
    return rows, dict(shard_counts)


def _mirrored_seed_scores(
    seat_scores: dict[tuple[int, str], float], *, condition: str
) -> dict[int, float]:
    """Average the two correlated seat readings into one seed-paired score."""

    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for (seed, seat), score in seat_scores.items():
        by_seed[seed][seat] = score
    missing = {
        seed: sorted(set(SEATS) - set(scores))
        for seed, scores in by_seed.items()
        if set(scores) != set(SEATS)
    }
    if missing:
        raise SystemExit(
            f"{condition}: calibration rows do not contain both seats for every seed: "
            f"{sorted(missing.items())[:3]}"
        )
    return {
        seed: sum(by_seed[seed][seat] for seat in SEATS) / len(SEATS)
        for seed in sorted(by_seed)
    }


def report_for(shards: list[dict], *, calibration_id: str, layout: str, min_pairs: int) -> dict:
    metadata = [_calibration_metadata(shard) for shard in shards]
    _require_uniform_run_identity(shards, metadata)
    actual_id, actual_layout = metadata[0]["calibration_id"], metadata[0]["resource_layout"]
    if actual_id != calibration_id:
        raise SystemExit(
            f"expected calibration id {calibration_id!r}, found {actual_id!r}"
        )
    if actual_layout != layout:
        raise SystemExit(f"expected resource layout {layout!r}, found {actual_layout!r}")
    rows, shard_counts = _condition_rows(shards, metadata)
    seed_scores = {
        condition: _mirrored_seed_scores(scores, condition=condition)
        for condition, scores in rows.items()
    }
    if set(seed_scores["baseline"]) != set(seed_scores["reduced"]):
        raise SystemExit("baseline and reduced conditions do not cover the same seeds")
    keys = sorted(seed_scores["baseline"])
    if len(keys) < min_pairs:
        raise SystemExit(
            f"{len(keys)} mirrored seed pairs < required minimum {min_pairs}; refusing a "
            "partial calibration verdict"
        )
    from pokezero.mcts_eval.scoring import bootstrap_indices, bootstrap_paired_delta

    baseline = [seed_scores["baseline"][key] for key in keys]
    reduced = [seed_scores["reduced"][key] for key in keys]
    interval = bootstrap_paired_delta(
        reduced,
        baseline,
        bootstrap_indices(
            sample_size=len(keys), resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
        ),
    )
    deltas = [
        reduced_score - baseline_score
        for reduced_score, baseline_score in zip(reduced, baseline, strict=True)
    ]
    checkpoint = shards[0]["checkpoint"]
    return {
        "schema_version": SCHEMA_VERSION,
        "calibration": {
            "id": actual_id,
            "resource_layout": actual_layout,
            "checkpoint": checkpoint,
            "checkpoint_sha256": shards[0]["checkpoint_sha256"],
            "engine_fingerprint": shards[0]["engine_fingerprint"],
            "commit": shards[0]["commit"],
            "raw_policy": "same checkpoint in both conditions",
            "foulplay_budget_ms": {
                "baseline": FOULPLAY_BUDGET_CALIBRATION_BASELINE_MS,
                "reduced": FOULPLAY_BUDGET_CALIBRATION_REDUCED_MS,
                "cut_fraction": FOULPLAY_BUDGET_CALIBRATION_CUT_FRACTION,
            },
            "thread_pin": dict(sorted(THREAD_PIN_ENV.items())),
        },
        "conditions": {
            condition: {
                "shards": shard_counts[condition],
                "mirrored_seed_pairs": len(seed_scores[condition]),
                "seat_observations": len(rows[condition]),
                "score_rate": (
                    sum(seed_scores[condition].values()) / len(seed_scores[condition])
                ),
            }
            for condition in sorted(rows)
        },
        "paired_comparison": {
            "estimand": "raw PokeZero score at FoulPlay 760 ms minus score at 1000 ms",
            "positive_direction": "the lower FoulPlay budget raised raw PokeZero score",
            "mirrored_seed_pairs": len(keys),
            "reduced_minus_baseline": interval.to_payload(),
            "discordant": {
                "reduced_higher_score": sum(delta > 0 for delta in deltas),
                "baseline_higher_score": sum(delta < 0 for delta in deltas),
                "tied_score": sum(delta == 0 for delta in deltas),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--resource-layout", required=True)
    parser.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.min_pairs <= 0:
        raise SystemExit("--min-pairs must be positive")
    report = report_for(
        load_shards(args.shards),
        calibration_id=args.calibration_id,
        layout=args.resource_layout,
        min_pairs=args.min_pairs,
    )
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["paired_comparison"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
