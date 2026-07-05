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
  watchdog           --run-id ID [--timeline FILE]  (default: stdin)
                     Ecology watchdogs over eval-timeline.jsonl rows. Prints one
                     alarm/warning JSON object per line (empty output = healthy).
  record             Assemble one ledger JSONL line for a probed (or skipped)
                     milestone and append it. Idempotent; the check-then-append
                     runs under an exclusive flock on the ledger file.
  lock               --fd N: take a non-blocking exclusive flock on inherited
                     file descriptor N (exit 1 if already held). The caller keeps
                     the fd open so the lock lives for the caller's lifetime.
  pools              --repo DIR [--format tsv|json]
                     Resolve the value-calibration eval pools from
                     POKEZERO_POOL_SELF / POKEZERO_POOL_FP (falling back to the
                     frozen v1 defaults) and print one TSV row per pool
                     (role, path, label, source) plus "# WARNING:" comment
                     lines. A pool's label keys its ledger entries and pearson
                     output filenames. Resolving to a v1 pool emits a
                     deprecation warning: v1 pools store v1-encoded
                     observations that the v2 schema guards refuse, so they
                     cannot score obsv2 checkpoints.

Watchdog thresholds (the 4L lesson — see evaluate_watchdogs). All alarms fire
strictly beyond their threshold (exactly-at-threshold is quiet):
  game-length drift        latest low-fi avg_game_length > 1.5x baseline (+50%).
                           Baseline: the run's own low-fi rows in the 30k-100k
                           band when present, else the earliest rows in the
                           timeline window (continuation runs / truncated tails),
                           else a LOUD "watchdog degraded" warning — never silence.
  strength regression      latest max-damage win rate more than 10 points
                           (absolute) below the run's own peak at matched fidelity
  policy-entropy floor     latest policy_entropy < 0.35
Non-finite (NaN/Infinity) watched values are quarantined before the math and
surface as a timeline_data_quality warning; rows without a usable
completed_games sort last (they are plausibly the newest, file-tail rows).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import math
import os
import re
import sys
import textwrap
from pathlib import Path

SCHEMA_VERSION = "pokezero.milestone_probe.v1"
ALERT_SCHEMA_VERSION = "pokezero.ecology_alert.v1"

POOL_SELF_ENV = "POKEZERO_POOL_SELF"
POOL_FP_ENV = "POKEZERO_POOL_FP"
# Default eval pools, repo-relative. Still the frozen v1 pools: switch these
# to runs/pool-self-v2-20260705/ and runs/pool-fp-v2-20260705/ once those
# pools exist and their READMEs say FROZEN (they are built from the 50k obsv2
# checkpoint). Until then obsv2 runs must override via the env vars above —
# the v2 schema guards refuse the v1-encoded observations stored here.
DEFAULT_POOL_SELF = "runs/e1-value-readiness-20260703/belief-1-5m/heldout-rollouts.jsonl"
DEFAULT_POOL_FP = "runs/pool-fp-v1-20260704/pool-fp-v1.jsonl"
# Historical ledger/filename labels for the v1 defaults (the self pool's
# filename stem, heldout-rollouts, never was its label).
V1_POOL_LABELS = {
    DEFAULT_POOL_SELF: "pool-self-v1",
    DEFAULT_POOL_FP: "pool-fp-v1",
}
V2_POOL_HINTS = {
    "self": "runs/pool-self-v2-20260705/",
    "fp": "runs/pool-fp-v2-20260705/",
}

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
        watched = [row.get("completed_games")]
        collapse = row.get("collapse")
        if isinstance(collapse, dict):
            watched += [collapse.get("avg_game_length"), collapse.get("policy_entropy")]
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("max-damage"), dict):
            watched.append(metrics["max-damage"].get("win_rate"))
        if any(value is not None and _finite(value) is None for value in watched):
            bad += 1
    return bad


def evaluate_watchdogs(
    rows: list[dict],
    *,
    drift_ratio: float = GAME_LENGTH_DRIFT_RATIO,
    regression_points: float = STRENGTH_REGRESSION_POINTS,
    entropy_floor: float = POLICY_ENTROPY_FLOOR,
    baseline_band: tuple[int, int] = BASELINE_BAND_GAMES,
    fallback_baseline_rows: int = DRIFT_FALLBACK_BASELINE_ROWS,
) -> list[dict]:
    """Ecology watchdogs over eval-timeline.jsonl rows (dicts, file order).

    Returns a list of alarm/warning dicts (empty = healthy). Pure function: no
    I/O. Every alarm fires strictly beyond its threshold. Non-finite watched
    values are quarantined first and reported via a timeline_data_quality
    warning so a NaN-emitting trainer cannot silently mute the dogs.

      game_length_drift    latest low-fi collapse.avg_game_length vs a baseline
                           mean. Baseline preference order:
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
      strength_regression  latest max-damage win_rate vs the run's own earlier
                           peak at the SAME fidelity (low-fi 600-game and high-fi
                           2000-game reads are not comparable). Alarms when the
                           drop strictly exceeds regression_points, with a float
                           epsilon so 0.85->0.75 behaves exactly like
                           0.80->0.70.
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
        latest_milestone = latest.get("milestone_games") or 0
        band_rows = [
            row
            for row in length_rows
            if baseline_band[0] <= (row.get("milestone_games") or 0) <= baseline_band[1]
        ]
        baseline_values: list[float] | None = None
        baseline_source = None
        if band_rows:
            if latest_milestone > baseline_band[1]:
                baseline_values = [
                    _finite(row["collapse"]["avg_game_length"]) for row in band_rows
                ]
                baseline_source = "band"
            # else: still establishing the band baseline — expected quiet
        elif len(length_rows) >= 2:
            earliest = length_rows[:-1][:fallback_baseline_rows]
            baseline_values = [
                _finite(row["collapse"]["avg_game_length"]) for row in earliest
            ]
            baseline_source = "earliest_rows"
        else:
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
                        f"{baseline_band[1] // 1000}k band and too few rows in the "
                        "timeline window for a relative baseline)"
                    ),
                }
            )
        if baseline_values:
            baseline_mean = sum(baseline_values) / len(baseline_values)
            if baseline_mean > 0 and latest_length - drift_ratio * baseline_mean > _EPSILON:
                alarms.append(
                    {
                        "watchdog": "game_length_drift",
                        "severity": "alarm",
                        "milestone_games": latest_milestone,
                        "completed_games": latest.get("completed_games"),
                        "latest_avg_game_length": latest_length,
                        "baseline_avg_game_length": round(baseline_mean, 4),
                        "baseline_source": baseline_source,
                        "baseline_rows": len(baseline_values),
                        "threshold_ratio": drift_ratio,
                        "message": (
                            f"avg_game_length {latest_length:.1f} is "
                            f"{latest_length / baseline_mean:.2f}x the "
                            f"{baseline_source} baseline {baseline_mean:.1f} "
                            f"(alarm ratio {drift_ratio})"
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
            peak_row = max(
                earlier, key=lambda row: _finite(row["metrics"]["max-damage"]["win_rate"])
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
# eval-pool resolution (pure function over env — unit-tested)
# ---------------------------------------------------------------------------


def _pool_label(path: str) -> str:
    """Ledger/filename label for a pool file: its sanitized filename stem."""
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(path).stem).strip("-._")
    if not label:
        raise SystemExit(f"cannot derive a pool label from {path!r}")
    return label


def resolve_pools(repo_root: Path | str, env) -> dict:
    """Resolve the value-calibration eval pools from the environment.

    POKEZERO_POOL_SELF / POKEZERO_POOL_FP override the frozen v1 defaults
    (empty values count as unset, matching shell ${VAR:-default} semantics);
    relative and ~ paths resolve against the repo root / home. No filesystem
    access — existence checks stay in the caller.

    Returns {"self": {path, label, source}, "fp": {...}, "warnings": [...]}.
    A pool resolving to a v1 default carries a deprecation warning whatever
    its source: v1 pools store v1-encoded observations that the v2 schema
    guards refuse, so every Pearson probe of an obsv2 checkpoint against
    them fails.
    """
    repo_root = Path(repo_root)
    pools: dict = {"warnings": []}
    for role, env_var, default in (
        ("self", POOL_SELF_ENV, DEFAULT_POOL_SELF),
        ("fp", POOL_FP_ENV, DEFAULT_POOL_FP),
    ):
        raw = env.get(env_var) or default
        source = "env" if raw is not default else "default"
        path = Path(os.path.expanduser(raw))
        if not path.is_absolute():
            path = repo_root / path
        # v1 pools keep their historical labels so existing ledgers/pearson
        # files stay valid even when the path arrives via the env var.
        v1_default = path == repo_root / default
        label = V1_POOL_LABELS[default] if v1_default else _pool_label(raw)
        pools[role] = {"path": str(path), "label": label, "source": source}
        if v1_default:
            pools["warnings"].append(
                f"DEPRECATED {role} pool: {label} ({path}) is v1-encoded; the v2 "
                f"schema guards refuse it, so it CANNOT score obsv2 checkpoints "
                f"and every ~100k milestone Pearson probe on an obsv2 run will "
                f"fail. Set {env_var} to the frozen v2 pool "
                f"(expected under {V2_POOL_HINTS[role]}) for obsv2 runs; the "
                f"defaults switch to v2 once those pools are frozen."
            )
    if pools["self"]["label"] == pools["fp"]["label"]:
        raise SystemExit(
            f"self and fp pools resolve to the same label "
            f"{pools['self']['label']!r} — their ledger entries and pearson "
            f"files would collide; rename one pool file"
        )
    return pools


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


def _cmd_pools(args: argparse.Namespace) -> int:
    pools = resolve_pools(Path(args.repo), os.environ)
    if args.format == "json":
        print(json.dumps(pools, indent=2, sort_keys=True))
        return 0
    # tsv for the bash config block: "# WARNING:" comments, then one row per
    # pool (role, path, label, source).
    for warning in pools["warnings"]:
        for line in textwrap.wrap(
            warning, width=96, break_long_words=False, break_on_hyphens=False
        ):
            print(f"# WARNING: {line}")
    for role in ("self", "fp"):
        pool = pools[role]
        print(f"{role}\t{pool['path']}\t{pool['label']}\t{pool['source']}")
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

    pools = sub.add_parser("pools")
    pools.add_argument("--repo", required=True, help="repo root for the repo-relative default pools")
    pools.add_argument("--format", choices=("tsv", "json"), default="tsv")
    pools.set_defaults(func=_cmd_pools)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
