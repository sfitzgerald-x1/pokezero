#!/usr/bin/env python3
"""JSON/logic helper for scripts/milestone_probes.sh (WS-3 items 2-3 of
docs/next_train_readiness_plan.md). Stdlib only — invoked with plain python3.

Subcommands:
  run-id-from-args   stdin: pod container args as a JSON array -> prints the
                     value following "--run-id".
  plan               --status-json FILE --ledger FILE [--step N] [--format json|tsv]
                     Diff a run's STATUS.json against the local milestone ledger
                     and emit the pending ~100k milestones, each mapped to the
                     milestone-nearest completed iteration checkpoint. Milestones
                     whose nearest checkpoint is more than --max-distance games
                     away (checkpoint rotation, retention gaps) come back as
                     action=skip so the sweep records a SKIPPED ledger line
                     instead of probing the wrong checkpoint.
  watchdog           --run-id ID [--timeline FILE] [--baseline-file FILE]
                     [--no-persist]  (timeline default: stdin)
                     Ecology watchdogs over eval-timeline.jsonl rows. Prints one
                     alarm/warning JSON object per line (empty output = healthy).
                     --baseline-file persists the first-computed game-length
                     drift baseline (read when present, written once) so the
                     baseline cannot slide forward with the timeline tail
                     window; --no-persist reads but never writes it (dry-run).
  record             Assemble one ledger JSONL line for a probed (or skipped)
                     milestone and append it. Idempotent; the check-then-append
                     runs under an exclusive flock on the ledger file.
  lock               --fd N: take a non-blocking exclusive flock on inherited
                     file descriptor N (exit 1 if already held). The caller keeps
                     the fd open so the lock lives for the caller's lifetime.

Watchdog thresholds (the 4L lesson — see evaluate_watchdogs). All alarms fire
strictly beyond their threshold (exactly-at-threshold is quiet):
  game-length drift        latest low-fi avg_game_length > 1.5x baseline (+50%).
                           Baseline: the per-run baseline persisted at first
                           computation (--baseline-file — immune to the tail
                           window sliding forward), else the run's own low-fi
                           rows in the 30k-100k band, else the earliest rows in
                           the timeline window (continuation runs / truncated
                           tails), else a LOUD "watchdog degraded" warning —
                           never silence.
  strength regression      the latest max-damage win rate OF EACH fidelity more
                           than 10 points (absolute) below that fidelity's own
                           earlier peak (a newer healthy high-fi row must not
                           mask a low-fi collapse)
  policy-entropy floor     latest policy_entropy < 0.35
Non-finite (NaN/Infinity) watched values — completed_games, milestone_games,
avg_game_length, policy_entropy, max-damage win_rate — are quarantined before
the math and surface as a timeline_data_quality warning; rows without a usable
completed_games sort last (they are plausibly the newest, file-tail rows).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import math
import os
import sys
from pathlib import Path

SCHEMA_VERSION = "pokezero.milestone_probe.v1"
ALERT_SCHEMA_VERSION = "pokezero.ecology_alert.v1"
BASELINE_SCHEMA_VERSION = "pokezero.drift_baseline.v1"

MILESTONE_STEP = 100_000
MILESTONE_MAX_DISTANCE_GAMES = 30_000
BASELINE_BAND_GAMES = (30_000, 100_000)
DRIFT_FALLBACK_BASELINE_ROWS = 5
GAME_LENGTH_DRIFT_RATIO = 1.5
STRENGTH_REGRESSION_POINTS = 0.10
POLICY_ENTROPY_FLOOR = 0.35
_EPSILON = 1e-9


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# watchdogs (pure functions over timeline rows — unit-tested)
# ---------------------------------------------------------------------------


def _finite(value) -> float | None:
    """The value as a float when it is a finite number; None otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _sorted_rows(rows: list) -> list[dict]:
    """Rows sorted by completed_games; rows without a finite completed_games
    sort LAST in their original order (append-only file tail: plausibly newest),
    never first where they would mask the true latest read."""
    dicts = [row for row in rows if isinstance(row, dict)]
    present = [row for row in dicts if _finite(row.get("completed_games")) is not None]
    missing = [row for row in dicts if _finite(row.get("completed_games")) is None]
    present.sort(key=lambda row: _finite(row.get("completed_games")))
    return present + missing


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
        if isinstance(max_damage, dict) and _finite(max_damage.get("win_rate")) is not None:
            out.append(row)
    return out


def _count_non_finite_rows(rows: list[dict]) -> int:
    """Rows carrying a watched field that is present but not a finite number."""
    bad = 0
    for row in rows:
        watched = [row.get("completed_games"), row.get("milestone_games")]
        collapse = row.get("collapse")
        if isinstance(collapse, dict):
            watched += [collapse.get("avg_game_length"), collapse.get("policy_entropy")]
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("max-damage"), dict):
            watched.append(metrics["max-damage"].get("win_rate"))
        if any(value is not None and _finite(value) is None for value in watched):
            bad += 1
    return bad


def compute_drift_baseline(
    rows: list,
    *,
    baseline_band: tuple[int, int] = BASELINE_BAND_GAMES,
    fallback_baseline_rows: int = DRIFT_FALLBACK_BASELINE_ROWS,
) -> dict | None:
    """The game-length drift baseline the watchdog arms with over these rows.

    Returns {"avg_game_length": mean, "source": "band"|"earliest_rows",
    "rows": n}, or None while no baseline is usable: no length rows, the
    latest row still inside the band (baseline being established), a single
    length row, or a non-positive mean. Pure function — the CLI persists the
    first non-None result per run (drift-baseline.json) and reuses it on
    later sweeps, because both sources below are computed from the visible
    ``tail -n`` timeline window and would otherwise slide forward with it,
    letting drift slower than the alarm ratio per window escape unalarmed.
    """
    rows = _sorted_rows(rows)
    length_rows = [
        row
        for row in _collapse_rows(rows)
        if _finite(row["collapse"].get("avg_game_length")) is not None
    ]
    if not length_rows:
        return None
    latest_milestone = _finite(length_rows[-1].get("milestone_games")) or 0
    band_rows = [
        row
        for row in length_rows
        if baseline_band[0] <= (_finite(row.get("milestone_games")) or 0) <= baseline_band[1]
    ]
    if band_rows:
        if latest_milestone <= baseline_band[1]:
            return None  # still establishing the band baseline — expected quiet
        values = [_finite(row["collapse"]["avg_game_length"]) for row in band_rows]
        source = "band"
    elif len(length_rows) >= 2:
        values = [
            _finite(row["collapse"]["avg_game_length"])
            for row in length_rows[:-1][:fallback_baseline_rows]
        ]
        source = "earliest_rows"
    else:
        return None
    mean = sum(values) / len(values)
    if mean <= 0:
        return None
    return {"avg_game_length": mean, "source": source, "rows": len(values)}


def _usable_baseline(baseline) -> dict | None:
    """A normalized {avg_game_length, source, rows} baseline dict, or None
    when ``baseline`` is not usable (wrong shape, non-finite or non-positive
    mean) — an unusable persisted baseline must degrade to the window-computed
    one, never poison the math."""
    if not isinstance(baseline, dict):
        return None
    avg = _finite(baseline.get("avg_game_length"))
    if avg is None or avg <= 0:
        return None
    source = baseline.get("source")
    rows_count = baseline.get("rows")
    return {
        "avg_game_length": avg,
        "source": source if isinstance(source, str) and source else "persisted",
        "rows": rows_count
        if isinstance(rows_count, int) and not isinstance(rows_count, bool)
        else 0,
    }


def evaluate_watchdogs(
    rows: list[dict],
    *,
    drift_ratio: float = GAME_LENGTH_DRIFT_RATIO,
    regression_points: float = STRENGTH_REGRESSION_POINTS,
    entropy_floor: float = POLICY_ENTROPY_FLOOR,
    baseline_band: tuple[int, int] = BASELINE_BAND_GAMES,
    fallback_baseline_rows: int = DRIFT_FALLBACK_BASELINE_ROWS,
    persisted_baseline: dict | None = None,
) -> list[dict]:
    """Ecology watchdogs over eval-timeline.jsonl rows (dicts, file order).

    Returns a list of alarm/warning dicts (empty = healthy). Pure function: no
    I/O. Every alarm fires strictly beyond its threshold. Non-finite watched
    values are quarantined first and reported via a timeline_data_quality
    warning so a NaN-emitting trainer cannot silently mute the dogs.

      game_length_drift    latest low-fi collapse.avg_game_length vs a baseline
                           mean. Baseline preference order:
                             0. ``persisted_baseline`` — the baseline persisted
                                the first time this run armed the dog
                                (drift-baseline.json via --baseline-file). The
                                window-computed fallbacks below slide with the
                                ``tail -n`` timeline window; the persisted one
                                does not, so drift slower than the alarm ratio
                                per window cannot ratchet it forward. An
                                unusable value degrades to the fallbacks;
                             1. the run's own low-fi rows with milestone_games
                                inside ``baseline_band`` ("band") — quiet while
                                the run is still inside the band;
                             2. when the band is empty (continuation runs whose
                                rows carry absolute counts, or tails truncated
                                past the band): the earliest
                                ``fallback_baseline_rows`` low-fi rows in the
                                window, excluding the latest ("earliest_rows");
                             3. neither possible (a single length row): a
                                game_length_drift_no_baseline WARNING — the dog
                                degrades loudly, never silently.
                           Alarms when latest > drift_ratio * baseline.
      strength_regression  the latest max-damage win_rate of EACH fidelity vs
                           that fidelity's own earlier peak (low-fi 600-game
                           and high-fi 2000-game reads are not comparable, and
                           a newer healthy high-fi row must not mask a low-fi
                           collapse). One alarm per regressed fidelity; alarms
                           when the drop strictly exceeds regression_points,
                           with a float epsilon so 0.85->0.75 behaves exactly
                           like 0.80->0.70.
      policy_entropy_floor latest low-fi collapse.policy_entropy < entropy_floor.
    """
    rows = _sorted_rows(rows)
    alarms: list[dict] = []

    non_finite = _count_non_finite_rows(rows)
    if non_finite:
        alarms.append(
            {
                "watchdog": "timeline_data_quality",
                "severity": "warning",
                "non_finite_rows": non_finite,
                "message": (
                    f"{non_finite} timeline row(s) carry non-finite watched values "
                    f"(NaN/Infinity) — quarantined from watchdog math; a NaN-emitting "
                    f"trainer is itself collapse-adjacent"
                ),
            }
        )

    # -- game-length drift -------------------------------------------------
    collapse_rows = _collapse_rows(rows)
    length_rows = [
        row
        for row in collapse_rows
        if _finite(row["collapse"].get("avg_game_length")) is not None
    ]
    if length_rows:
        latest = length_rows[-1]
        latest_length = _finite(latest["collapse"]["avg_game_length"])
        latest_milestone = _finite(latest.get("milestone_games"))
        baseline = _usable_baseline(persisted_baseline)
        from_persisted = baseline is not None
        if baseline is None:
            baseline = compute_drift_baseline(
                rows,
                baseline_band=baseline_band,
                fallback_baseline_rows=fallback_baseline_rows,
            )
        if baseline is not None:
            baseline_mean = baseline["avg_game_length"]
            if latest_length - drift_ratio * baseline_mean > _EPSILON:
                alarms.append(
                    {
                        "watchdog": "game_length_drift",
                        "severity": "alarm",
                        "milestone_games": latest_milestone,
                        "completed_games": latest.get("completed_games"),
                        "latest_avg_game_length": latest_length,
                        "baseline_avg_game_length": round(baseline_mean, 4),
                        "baseline_source": baseline["source"],
                        "baseline_rows": baseline["rows"],
                        "baseline_persisted": from_persisted,
                        "threshold_ratio": drift_ratio,
                        "message": (
                            f"avg_game_length {latest_length:.1f} is "
                            f"{latest_length / baseline_mean:.2f}x the "
                            f"{baseline['source']} baseline {baseline_mean:.1f} "
                            f"(alarm ratio {drift_ratio})"
                        ),
                    }
                )
        elif not any(
            baseline_band[0] <= (_finite(row.get("milestone_games")) or 0) <= baseline_band[1]
            for row in length_rows
        ):
            # Band rows present with the run still inside the band is the
            # quiet "establishing" case; here the band is EMPTY and no
            # relative baseline was possible either — degrade loudly.
            alarms.append(
                {
                    "watchdog": "game_length_drift_no_baseline",
                    "severity": "warning",
                    "milestone_games": latest_milestone,
                    "completed_games": latest.get("completed_games"),
                    "latest_avg_game_length": latest_length,
                    "message": (
                        "no baseline — game-length drift watchdog degraded "
                        f"(no rows in the {baseline_band[0] // 1000}k-"
                        f"{baseline_band[1] // 1000}k band and no usable relative "
                        "baseline in the timeline window)"
                    ),
                }
            )

    # -- matched-fidelity strength regression --------------------------------
    # The latest read of EACH fidelity is checked against that fidelity's own
    # earlier peak: a newer healthy high-fi row must not mask a low-fi
    # collapse until the next low-fi row happens to arrive.
    strength_rows = _max_damage_rows(rows)
    rows_by_fidelity: dict = {}
    for row in strength_rows:
        rows_by_fidelity.setdefault(row.get("fidelity"), []).append(row)
    for fidelity, fidelity_rows in rows_by_fidelity.items():
        if len(fidelity_rows) < 2:
            continue
        latest = fidelity_rows[-1]
        peak_row = max(
            fidelity_rows[:-1],
            key=lambda row: _finite(row["metrics"]["max-damage"]["win_rate"]),
        )
        peak = _finite(peak_row["metrics"]["max-damage"]["win_rate"])
        latest_rate = _finite(latest["metrics"]["max-damage"]["win_rate"])
        if (peak - latest_rate) - regression_points > _EPSILON:
            alarms.append(
                {
                    "watchdog": "strength_regression",
                    "severity": "alarm",
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
                        f"({fidelity} fidelity, alarm beyond {regression_points * 100:.0f})"
                    ),
                }
            )

    # -- policy-entropy floor ------------------------------------------------
    entropy_rows = [
        row
        for row in collapse_rows
        if _finite(row["collapse"].get("policy_entropy")) is not None
    ]
    if entropy_rows:
        latest = entropy_rows[-1]
        entropy = _finite(latest["collapse"]["policy_entropy"])
        if entropy_floor - entropy > _EPSILON:
            alarms.append(
                {
                    "watchdog": "policy_entropy_floor",
                    "severity": "alarm",
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


def _ledger_milestones_from_lines(lines) -> set[int]:
    milestones: set[int] = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        milestone = entry.get("milestone_games")
        if isinstance(milestone, bool):
            continue
        if isinstance(milestone, int):
            milestones.add(milestone)
        elif isinstance(milestone, float) and milestone.is_integer():
            milestones.add(int(milestone))
    return milestones


def _ledger_milestones(ledger_path: Path) -> set[int]:
    if not ledger_path.exists():
        return set()
    return _ledger_milestones_from_lines(
        ledger_path.read_text(encoding="utf-8").splitlines()
    )


def pending_milestones(
    status: dict,
    done: set[int],
    *,
    step: int = MILESTONE_STEP,
    max_distance: int = MILESTONE_MAX_DISTANCE_GAMES,
) -> dict:
    """Map a run STATUS.json to the not-yet-probed ~``step`` milestones.

    Milestone m (m = step, 2*step, ... <= completed games) is mapped to the
    completed iteration whose cumulative game count is nearest to m. When the
    nearest checkpoint is more than ``max_distance`` games away (checkpoint
    rotation, retention gaps) the item comes back with action="skip" and a
    skip_reason — probing a checkpoint that far from the milestone would be
    silently wrong science. Pure function: no I/O.
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
            games_at = started_from + nearest["iteration"] * gpi
            distance = abs(games_at - milestone)
            item = {
                "milestone_games": milestone,
                "iteration": nearest["iteration"],
                "games_at_iteration": games_at,
                "distance_games": distance,
                "remote_checkpoint": nearest["checkpoint_path"],
                "local_name": f"{run_id}-i{nearest['iteration']}.pt",
                "action": "probe",
                "skip_reason": None,
            }
            if distance > max_distance:
                item["action"] = "skip"
                item["skip_reason"] = (
                    f"nearest checkpoint (iteration {nearest['iteration']}, "
                    f"{games_at} games) is {distance} games from milestone "
                    f"{milestone} (max {max_distance})"
                )
            pending.append(item)
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
    plan = pending_milestones(status, done, step=args.step, max_distance=args.max_distance)
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
                f"{item['games_at_iteration']}\t{item['distance_games']}\t"
                f"{item['action']}\t{item['remote_checkpoint']}\t{item['local_name']}\t"
                f"{item['skip_reason'] or ''}"
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

    baseline_path = Path(args.baseline_file) if args.baseline_file else None
    persisted = None
    alarms: list[dict] = []
    if baseline_path is not None and baseline_path.exists():
        try:
            raw = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = None
        persisted = _usable_baseline(raw)
        if persisted is None:
            # Never overwrite the bad file: a quiet rewrite would reset the
            # baseline to the current window — the ratchet this file exists to
            # prevent. Warn every sweep until an operator fixes or removes it.
            alarms.append(
                {
                    "watchdog": "drift_baseline_unreadable",
                    "severity": "warning",
                    "baseline_file": str(baseline_path),
                    "message": (
                        f"persisted drift baseline {baseline_path} is unreadable or "
                        "invalid — falling back to the window-relative baseline; "
                        "left in place for inspection: fix or remove it"
                    ),
                }
            )

    alarms += evaluate_watchdogs(rows, persisted_baseline=persisted)

    if baseline_path is not None and not baseline_path.exists() and not args.no_persist:
        fresh = compute_drift_baseline(rows)
        if fresh is not None:
            payload = {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "run_id": args.run_id,
                "computed_at_utc": _utc_now(),
                **fresh,
            }
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = baseline_path.with_name(baseline_path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp_path, baseline_path)
            print(
                f"[watchdog] drift baseline persisted -> {baseline_path} "
                f"({fresh['source']}, avg_game_length {fresh['avg_game_length']:.2f})",
                file=sys.stderr,
            )

    for alarm in alarms:
        alarm = {
            "schema_version": ALERT_SCHEMA_VERSION,
            "recorded_at_utc": _utc_now(),
            "run_id": args.run_id,
            **alarm,
        }
        print(json.dumps(alarm, sort_keys=True))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    entry: dict = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": _utc_now(),
        "run_id": args.run_id,
        "milestone_games": args.milestone,
        "iteration": args.iteration,
        "checkpoint": args.checkpoint,
    }
    if args.games_at is not None:
        entry["games_at_iteration"] = args.games_at
        entry["distance_games"] = abs(args.games_at - args.milestone)
    if args.skip_reason:
        entry["skipped"] = True
        entry["skip_reason"] = args.skip_reason
    else:
        entry["pools"] = {}
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

    # Exclusive flock around the check-then-append so concurrent recorders
    # (belt-and-braces under the sweep-level lock) cannot duplicate a milestone.
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            if args.milestone in _ledger_milestones_from_lines(handle):
                print(
                    f"[record] milestone {args.milestone} already in {ledger_path}; skipping",
                    file=sys.stderr,
                )
                return 0
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    verb = "SKIPPED" if args.skip_reason else "recorded"
    print(f"[record] {args.run_id} milestone {args.milestone} {verb} -> {ledger_path}", file=sys.stderr)
    return 0


def _cmd_lock(args: argparse.Namespace) -> int:
    try:
        fcntl.flock(args.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run-id-from-args").set_defaults(func=_cmd_run_id_from_args)

    plan = sub.add_parser("plan")
    plan.add_argument("--status-json", required=True)
    plan.add_argument("--ledger", required=True)
    plan.add_argument("--step", type=int, default=MILESTONE_STEP)
    plan.add_argument("--max-distance", type=int, default=MILESTONE_MAX_DISTANCE_GAMES)
    plan.add_argument("--format", choices=("json", "tsv"), default="json")
    plan.set_defaults(func=_cmd_plan)

    watchdog = sub.add_parser("watchdog")
    watchdog.add_argument("--run-id", required=True)
    watchdog.add_argument("--timeline", default=None, help="eval-timeline.jsonl (default: stdin)")
    watchdog.add_argument(
        "--baseline-file",
        default=None,
        help=(
            "per-run drift-baseline JSON (runs/milestone-probes/<run>/drift-baseline.json): "
            "read when present so the drift baseline cannot slide with the timeline "
            "window; written once, the first time a baseline is computed"
        ),
    )
    watchdog.add_argument(
        "--no-persist",
        action="store_true",
        help="never write --baseline-file (dry-run); reading is unaffected",
    )
    watchdog.set_defaults(func=_cmd_watchdog)

    record = sub.add_parser("record")
    record.add_argument("--ledger", required=True)
    record.add_argument("--run-id", required=True)
    record.add_argument("--milestone", type=int, required=True)
    record.add_argument("--iteration", type=int, required=True)
    record.add_argument("--checkpoint", required=True)
    record.add_argument("--games-at", type=int, default=None, help="cumulative games at the mapped iteration")
    record.add_argument(
        "--pearson",
        action="append",
        metavar="POOL=PATH",
        help="value-calibration --json output for one eval pool (repeatable)",
    )
    record.add_argument("--hazard", default=None, help="hazard_probe.py --out JSON")
    record.add_argument("--hazard-label", default=None)
    record.add_argument(
        "--skip-reason",
        default=None,
        help="record the milestone as SKIPPED (no probe metrics) with this reason",
    )
    record.set_defaults(func=_cmd_record)

    lock = sub.add_parser("lock")
    lock.add_argument("--fd", type=int, required=True, help="inherited fd to flock (LOCK_EX|LOCK_NB)")
    lock.set_defaults(func=_cmd_lock)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
