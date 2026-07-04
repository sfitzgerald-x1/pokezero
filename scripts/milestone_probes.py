#!/usr/bin/env python3
"""JSON/logic helper for scripts/milestone_probes.sh (WS-3 items 2-3 of
docs/next_train_readiness_plan.md). Stdlib only — invoked with plain python3.

Subcommands:
  run-id-from-args   stdin: pod container args as a JSON array -> prints the
                     value following "--run-id".
  plan               --status-json FILE --ledger FILE [--step N] [--format json|tsv]
                     Diff a run's STATUS.json against the local milestone ledger
                     and emit the pending ~100k milestones, each mapped to the
                     milestone-nearest completed iteration checkpoint.
  watchdog           --run-id ID [--timeline FILE]  (default: stdin)
                     Ecology watchdogs over eval-timeline.jsonl rows. Prints one
                     alarm JSON object per line (empty output = healthy).
  record             Assemble one ledger JSONL line for a probed milestone from
                     the probe output files and append it (idempotent).

Watchdog thresholds (the 4L lesson — see evaluate_watchdogs):
  game-length drift        latest low-fi avg_game_length > 1.5x the run's own
                           30k-100k baseline mean (+50%)
  strength regression      latest max-damage win rate >= 10 points (absolute)
                           below the run's own peak at matched fidelity
  policy-entropy floor     latest policy_entropy < 0.35
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "pokezero.milestone_probe.v1"
ALERT_SCHEMA_VERSION = "pokezero.ecology_alert.v1"

MILESTONE_STEP = 100_000
BASELINE_BAND_GAMES = (30_000, 100_000)
GAME_LENGTH_DRIFT_RATIO = 1.5
STRENGTH_REGRESSION_POINTS = 0.10
POLICY_ENTROPY_FLOOR = 0.35


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# watchdogs (pure functions over timeline rows — unit-tested)
# ---------------------------------------------------------------------------


def _collapse_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("fidelity") == "low"
        and isinstance(row.get("collapse"), dict)
    ]


def _max_damage_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        max_damage = metrics.get("max-damage")
        if isinstance(max_damage, dict) and max_damage.get("win_rate") is not None:
            out.append(row)
    return out


def evaluate_watchdogs(
    rows: list[dict],
    *,
    drift_ratio: float = GAME_LENGTH_DRIFT_RATIO,
    regression_points: float = STRENGTH_REGRESSION_POINTS,
    entropy_floor: float = POLICY_ENTROPY_FLOOR,
    baseline_band: tuple[int, int] = BASELINE_BAND_GAMES,
) -> list[dict]:
    """Ecology watchdogs over eval-timeline.jsonl rows (dicts, file order).

    Returns a list of alarm dicts (empty = healthy). Pure function: no I/O.

      game_length_drift    latest low-fi collapse.avg_game_length vs the mean of
                           the run's own low-fi rows with milestone_games inside
                           ``baseline_band``. Alarms when latest > drift_ratio *
                           baseline. Silent while the run is still inside the
                           baseline band or has no baseline rows yet.
      strength_regression  latest max-damage win_rate vs the run's own earlier
                           peak at the SAME fidelity (low-fi 600-game and high-fi
                           2000-game reads are not comparable). Alarms when
                           peak - latest >= regression_points.
      policy_entropy_floor latest low-fi collapse.policy_entropy < entropy_floor.
    """
    rows = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: (row.get("completed_games") or 0),
    )
    alarms: list[dict] = []

    # -- game-length drift -------------------------------------------------
    collapse_rows = _collapse_rows(rows)
    length_rows = [
        row for row in collapse_rows if row["collapse"].get("avg_game_length") is not None
    ]
    if length_rows:
        latest = length_rows[-1]
        baseline = [
            row["collapse"]["avg_game_length"]
            for row in length_rows
            if baseline_band[0] <= (row.get("milestone_games") or 0) <= baseline_band[1]
        ]
        latest_milestone = latest.get("milestone_games") or 0
        if baseline and latest_milestone > baseline_band[1]:
            baseline_mean = sum(baseline) / len(baseline)
            latest_length = latest["collapse"]["avg_game_length"]
            if baseline_mean > 0 and latest_length > drift_ratio * baseline_mean:
                alarms.append(
                    {
                        "watchdog": "game_length_drift",
                        "milestone_games": latest_milestone,
                        "completed_games": latest.get("completed_games"),
                        "latest_avg_game_length": latest_length,
                        "baseline_avg_game_length": round(baseline_mean, 4),
                        "baseline_band_games": list(baseline_band),
                        "threshold_ratio": drift_ratio,
                        "message": (
                            f"avg_game_length {latest_length:.1f} is "
                            f"{latest_length / baseline_mean:.2f}x the "
                            f"{baseline_band[0] // 1000}k-{baseline_band[1] // 1000}k "
                            f"baseline {baseline_mean:.1f} (alarm ratio {drift_ratio})"
                        ),
                    }
                )

    # -- matched-milestone strength regression ------------------------------
    strength_rows = _max_damage_rows(rows)
    if len(strength_rows) >= 2:
        latest = strength_rows[-1]
        fidelity = latest.get("fidelity")
        earlier = [
            row for row in strength_rows[:-1] if row.get("fidelity") == fidelity
        ]
        if earlier:
            peak_row = max(earlier, key=lambda row: row["metrics"]["max-damage"]["win_rate"])
            peak = peak_row["metrics"]["max-damage"]["win_rate"]
            latest_rate = latest["metrics"]["max-damage"]["win_rate"]
            if peak - latest_rate >= regression_points:
                alarms.append(
                    {
                        "watchdog": "strength_regression",
                        "milestone_games": latest.get("milestone_games"),
                        "completed_games": latest.get("completed_games"),
                        "fidelity": fidelity,
                        "latest_max_damage_win_rate": latest_rate,
                        "peak_max_damage_win_rate": peak,
                        "peak_milestone_games": peak_row.get("milestone_games"),
                        "threshold_points": regression_points,
                        "message": (
                            f"max-damage win rate {latest_rate:.3f} is "
                            f"{(peak - latest_rate) * 100:.1f} points below the run peak "
                            f"{peak:.3f} @ {peak_row.get('milestone_games')} games "
                            f"({fidelity} fidelity, alarm at {regression_points * 100:.0f})"
                        ),
                    }
                )

    # -- policy-entropy floor ------------------------------------------------
    entropy_rows = [
        row for row in _collapse_rows(rows) if row["collapse"].get("policy_entropy") is not None
    ]
    if entropy_rows:
        latest = entropy_rows[-1]
        entropy = latest["collapse"]["policy_entropy"]
        if entropy < entropy_floor:
            alarms.append(
                {
                    "watchdog": "policy_entropy_floor",
                    "milestone_games": latest.get("milestone_games"),
                    "completed_games": latest.get("completed_games"),
                    "policy_entropy": entropy,
                    "threshold": entropy_floor,
                    "message": (
                        f"policy_entropy {entropy:.4f} < {entropy_floor} — collapse risk"
                    ),
                }
            )

    return alarms


# ---------------------------------------------------------------------------
# milestone planning (STATUS.json -> pending milestones)
# ---------------------------------------------------------------------------


def _ledger_milestones(ledger_path: Path) -> set[int]:
    milestones: set[int] = set()
    if not ledger_path.exists():
        return milestones
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        milestone = entry.get("milestone_games")
        if isinstance(milestone, int):
            milestones.add(milestone)
    return milestones


def pending_milestones(
    status: dict,
    done: set[int],
    *,
    step: int = MILESTONE_STEP,
) -> dict:
    """Map a run STATUS.json to the not-yet-probed ~``step`` milestones.

    Milestone m (m = step, 2*step, ... <= completed games) is mapped to the
    completed iteration whose cumulative game count is nearest to m. Pure
    function: no I/O.
    """
    run_id = status.get("run_id")
    gpi = status.get("games_per_iteration") or 0
    started_from = status.get("started_from_completed_games") or 0
    iterations = [
        it
        for it in status.get("completed_iterations", [])
        if isinstance(it.get("iteration"), int) and it.get("checkpoint_path")
    ]
    if iterations and gpi > 0:
        completed_games = started_from + max(it["iteration"] for it in iterations) * gpi
    else:
        completed_games = started_from

    pending = []
    milestone = step
    while milestone <= completed_games:
        if milestone not in done and milestone > started_from and iterations:
            nearest = min(
                iterations,
                key=lambda it: abs(started_from + it["iteration"] * gpi - milestone),
            )
            pending.append(
                {
                    "milestone_games": milestone,
                    "iteration": nearest["iteration"],
                    "games_at_iteration": started_from + nearest["iteration"] * gpi,
                    "remote_checkpoint": nearest["checkpoint_path"],
                    "local_name": f"{run_id}-i{nearest['iteration']}.pt",
                }
            )
        milestone += step

    return {
        "run_id": run_id,
        "status": status.get("status"),
        "completed_games": completed_games,
        "games_per_iteration": gpi,
        "latest_checkpoint_path": status.get("latest_checkpoint_path"),
        "pending": pending,
    }


# ---------------------------------------------------------------------------
# ledger recording
# ---------------------------------------------------------------------------

_SCALAR_TYPES = (int, float, str, bool, type(None))


def _scalars(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if isinstance(value, _SCALAR_TYPES)}


def _hazard_row(payload: dict, label: str | None) -> dict:
    rows = payload.get("checkpoints") or []
    if label is not None:
        for row in rows:
            if row.get("label") == label:
                return _scalars(row)
    if len(rows) == 1:
        return _scalars(rows[0])
    raise SystemExit(f"hazard payload has {len(rows)} checkpoint rows; pass --hazard-label")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_run_id_from_args(_args: argparse.Namespace) -> int:
    pod_args = json.load(sys.stdin)
    for index, value in enumerate(pod_args):
        if value == "--run-id" and index + 1 < len(pod_args):
            print(pod_args[index + 1])
            return 0
    print("no --run-id in pod args", file=sys.stderr)
    return 1


def _cmd_plan(args: argparse.Namespace) -> int:
    status = json.loads(Path(args.status_json).read_text(encoding="utf-8"))
    done = _ledger_milestones(Path(args.ledger))
    plan = pending_milestones(status, done, step=args.step)
    if args.format == "json":
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:  # tsv for the bash loop: comment header, then one row per milestone
        print(
            f"# run={plan['run_id']} status={plan['status']} "
            f"completed_games={plan['completed_games']} pending={len(plan['pending'])} "
            f"already_probed={len(done & set(range(args.step, plan['completed_games'] + 1, args.step)))}"
        )
        for item in plan["pending"]:
            print(
                f"{item['milestone_games']}\t{item['iteration']}\t"
                f"{item['games_at_iteration']}\t{item['remote_checkpoint']}\t{item['local_name']}"
            )
    return 0


def _cmd_watchdog(args: argparse.Namespace) -> int:
    stream = open(args.timeline, encoding="utf-8") if args.timeline else sys.stdin
    rows = []
    with stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for alarm in evaluate_watchdogs(rows):
        alarm = {
            "schema_version": ALERT_SCHEMA_VERSION,
            "recorded_at_utc": _utc_now(),
            "run_id": args.run_id,
            **alarm,
        }
        print(json.dumps(alarm, sort_keys=True))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger)
    if args.milestone in _ledger_milestones(ledger_path):
        print(
            f"[record] milestone {args.milestone} already in {ledger_path}; skipping",
            file=sys.stderr,
        )
        return 0
    entry: dict = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": _utc_now(),
        "run_id": args.run_id,
        "milestone_games": args.milestone,
        "iteration": args.iteration,
        "checkpoint": args.checkpoint,
        "pools": {},
    }
    for spec in args.pearson or []:
        pool, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--pearson expects POOL=PATH, got {spec!r}")
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        entry["pools"][pool] = _scalars(report)
    if args.hazard:
        payload = json.loads(Path(args.hazard).read_text(encoding="utf-8"))
        entry["hazard"] = _hazard_row(payload, args.hazard_label)
        entry["hazard"]["corpus_games"] = payload.get("corpus_games")
        entry["hazard"]["corpus_states"] = payload.get("corpus_states")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"[record] {args.run_id} milestone {args.milestone} -> {ledger_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run-id-from-args").set_defaults(func=_cmd_run_id_from_args)

    plan = sub.add_parser("plan")
    plan.add_argument("--status-json", required=True)
    plan.add_argument("--ledger", required=True)
    plan.add_argument("--step", type=int, default=MILESTONE_STEP)
    plan.add_argument("--format", choices=("json", "tsv"), default="json")
    plan.set_defaults(func=_cmd_plan)

    watchdog = sub.add_parser("watchdog")
    watchdog.add_argument("--run-id", required=True)
    watchdog.add_argument("--timeline", default=None, help="eval-timeline.jsonl (default: stdin)")
    watchdog.set_defaults(func=_cmd_watchdog)

    record = sub.add_parser("record")
    record.add_argument("--ledger", required=True)
    record.add_argument("--run-id", required=True)
    record.add_argument("--milestone", type=int, required=True)
    record.add_argument("--iteration", type=int, required=True)
    record.add_argument("--checkpoint", required=True)
    record.add_argument(
        "--pearson",
        action="append",
        metavar="POOL=PATH",
        help="value-calibration --json output for one eval pool (repeatable)",
    )
    record.add_argument("--hazard", default=None, help="hazard_probe.py --out JSON")
    record.add_argument("--hazard-label", default=None)
    record.set_defaults(func=_cmd_record)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
