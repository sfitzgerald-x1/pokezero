"""Unit tests for the ecology-watchdog, milestone-planning, and eval-pool
resolution logic in scripts/milestone_probes.py (WS-3 items 2-3, docs/next_train_readiness_plan.md).

The module is stdlib-only and lives outside the package, so it is loaded
straight from the scripts directory.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "milestone_probes.py"
_spec = importlib.util.spec_from_file_location("milestone_probes", _SCRIPT)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

_NO_COMPLETED = object()  # sentinel: omit completed_games entirely


def _row(
    milestone: int,
    *,
    fidelity: str = "low",
    avg_game_length: float | None = None,
    policy_entropy: float | None = None,
    max_damage: float | None = None,
    completed: int | object | None = None,
) -> dict:
    row = {
        "fidelity": fidelity,
        "milestone_games": milestone,
        "status": "running",
        "metrics": {},
    }
    if completed is not _NO_COMPLETED:
        row["completed_games"] = completed if completed is not None else milestone
    if max_damage is not None:
        row["metrics"]["max-damage"] = {"win_rate": max_damage, "wins": 0, "games": 600}
    if avg_game_length is not None or policy_entropy is not None:
        row["collapse"] = {"tie_rate": 0.004}
        if avg_game_length is not None:
            row["collapse"]["avg_game_length"] = avg_game_length
        if policy_entropy is not None:
            row["collapse"]["policy_entropy"] = policy_entropy
    return row


def _baseline_rows() -> list[dict]:
    """Healthy 30k-100k band: avg_game_length mean 28.0, entropy ~1.0."""
    return [
        _row(30_000, avg_game_length=27.0, policy_entropy=1.1, max_damage=0.30),
        _row(60_000, avg_game_length=28.0, policy_entropy=1.0, max_damage=0.45),
        _row(100_000, avg_game_length=29.0, policy_entropy=1.0, max_damage=0.55),
    ]


def _alarm_names(rows: list[dict]) -> set[str]:
    return {alarm["watchdog"] for alarm in mp.evaluate_watchdogs(rows)}


def _alarms_of(rows: list[dict], watchdog: str) -> list[dict]:
    return [a for a in mp.evaluate_watchdogs(rows) if a["watchdog"] == watchdog]


class TestGameLengthDrift:
    def test_healthy_run_no_alarms(self) -> None:
        rows = _baseline_rows() + [
            _row(200_000, avg_game_length=30.0, policy_entropy=0.9, max_damage=0.70),
        ]
        assert mp.evaluate_watchdogs(rows) == []

    def test_alarm_above_fifty_percent_drift(self) -> None:
        # baseline mean 28.0 -> alarm strictly above 42.0
        rows = _baseline_rows() + [_row(200_000, avg_game_length=42.1)]
        assert "game_length_drift" in _alarm_names(rows)

    def test_no_alarm_at_exact_threshold(self) -> None:
        rows = _baseline_rows() + [_row(200_000, avg_game_length=42.0)]
        assert "game_length_drift" not in _alarm_names(rows)

    def test_young_run_uses_relative_fallback_quietly(self) -> None:
        # inside 0-30k the band is empty; the earliest-rows fallback applies
        # and a healthy latest stays quiet
        rows = [
            _row(10_000, avg_game_length=30.0),
            _row(20_000, avg_game_length=31.0),
        ]
        assert "game_length_drift" not in _alarm_names(rows)
        assert "game_length_drift_no_baseline" not in _alarm_names(rows)

    def test_silent_while_inside_baseline_band(self) -> None:
        # band rows exist and the latest row is still inside the band:
        # the baseline is still being established
        rows = [
            _row(30_000, avg_game_length=20.0),
            _row(100_000, avg_game_length=200.0),
        ]
        assert "game_length_drift" not in _alarm_names(rows)

    def test_uses_latest_row_only(self) -> None:
        # a past spike that recovered must not alarm
        rows = _baseline_rows() + [
            _row(150_000, avg_game_length=80.0),
            _row(200_000, avg_game_length=29.0),
        ]
        assert "game_length_drift" not in _alarm_names(rows)

    def test_high_fidelity_rows_ignored(self) -> None:
        # collapse metrics ride the low-fi rows; a stray high-fi row with a
        # collapse dict must not become the "latest" read
        rows = _baseline_rows() + [
            _row(200_000, avg_game_length=29.0),
            _row(200_000, fidelity="high", avg_game_length=99.0, completed=200_100),
        ]
        assert "game_length_drift" not in _alarm_names(rows)

    def test_alarm_payload_fields(self) -> None:
        rows = _baseline_rows() + [_row(200_000, avg_game_length=56.0)]
        (alarm,) = _alarms_of(rows, "game_length_drift")
        assert alarm["latest_avg_game_length"] == 56.0
        assert alarm["baseline_avg_game_length"] == 28.0
        assert alarm["baseline_source"] == "band"
        assert alarm["milestone_games"] == 200_000
        assert alarm["threshold_ratio"] == mp.GAME_LENGTH_DRIFT_RATIO
        assert alarm["severity"] == "alarm"
        assert "message" in alarm


class TestGameLengthDriftContinuation:
    """H2: continuation runs / truncated tails never see the 30k-100k band —
    the dog must fall back to a relative baseline, not go silent."""

    def test_continuation_blowup_alarms(self) -> None:
        # rows carry absolute counts >= 600k (the 256d-1m shape); the 10x
        # blowup at 800k must alarm against the earliest-rows baseline
        rows = [
            _row(600_000, avg_game_length=28.0),
            _row(700_000, avg_game_length=29.0),
            _row(800_000, avg_game_length=280.0),
        ]
        (alarm,) = _alarms_of(rows, "game_length_drift")
        assert alarm["baseline_source"] == "earliest_rows"
        assert alarm["baseline_avg_game_length"] == 28.5
        assert alarm["latest_avg_game_length"] == 280.0

    def test_continuation_healthy_is_quiet(self) -> None:
        rows = [
            _row(600_000, avg_game_length=28.0),
            _row(700_000, avg_game_length=29.0),
            _row(800_000, avg_game_length=30.0),
        ]
        assert mp.evaluate_watchdogs(rows) == []

    def test_fallback_uses_earliest_rows_not_recent_ones(self) -> None:
        # 7 rows: baseline must be the FIRST 5, excluding the creeping recent
        # rows, so slow drift cannot ratchet its own baseline upward
        rows = [
            _row(600_000 + i * 10_000, avg_game_length=length)
            for i, length in enumerate([28.0, 28.0, 28.0, 28.0, 28.0, 40.0, 43.0])
        ]
        (alarm,) = _alarms_of(rows, "game_length_drift")
        assert alarm["baseline_avg_game_length"] == 28.0
        assert alarm["baseline_rows"] == 5

    def test_single_row_degrades_loudly_not_silently(self) -> None:
        # one length row: no baseline is possible — the dog must say so
        rows = [_row(600_000, avg_game_length=28.0)]
        (warning,) = _alarms_of(rows, "game_length_drift_no_baseline")
        assert warning["severity"] == "warning"
        assert "no baseline" in warning["message"]
        assert "game_length_drift" not in _alarm_names(rows)


class TestDriftBaselineCompute:
    """compute_drift_baseline: the pure baseline the CLI persists per run."""

    def test_band_baseline_once_beyond_band(self) -> None:
        rows = _baseline_rows() + [_row(200_000, avg_game_length=30.0)]
        assert mp.compute_drift_baseline(rows) == {
            "avg_game_length": 28.0,
            "source": "band",
            "rows": 3,
        }

    def test_none_while_still_inside_band(self) -> None:
        rows = [
            _row(30_000, avg_game_length=20.0),
            _row(100_000, avg_game_length=200.0),
        ]
        assert mp.compute_drift_baseline(rows) is None

    def test_none_for_empty_or_single_row(self) -> None:
        assert mp.compute_drift_baseline([]) is None
        assert mp.compute_drift_baseline([_row(600_000, avg_game_length=28.0)]) is None

    def test_earliest_rows_fallback_for_continuation(self) -> None:
        rows = [
            _row(600_000, avg_game_length=28.0),
            _row(700_000, avg_game_length=29.0),
            _row(800_000, avg_game_length=280.0),
        ]
        assert mp.compute_drift_baseline(rows) == {
            "avg_game_length": 28.5,
            "source": "earliest_rows",
            "rows": 2,
        }

    def test_fallback_caps_at_earliest_five(self) -> None:
        rows = [
            _row(600_000 + i * 10_000, avg_game_length=length)
            for i, length in enumerate([28.0, 28.0, 28.0, 28.0, 28.0, 40.0, 43.0])
        ]
        assert mp.compute_drift_baseline(rows) == {
            "avg_game_length": 28.0,
            "source": "earliest_rows",
            "rows": 5,
        }

    def test_non_positive_mean_is_unusable(self) -> None:
        rows = [
            _row(600_000, avg_game_length=0.0),
            _row(700_000, avg_game_length=0.0),
            _row(800_000, avg_game_length=0.0),
        ]
        assert mp.compute_drift_baseline(rows) is None


class TestPersistedBaseline:
    """Residual 1 of the #500 verify review: the earliest_rows fallback slides
    with the tail window, so drift slower than +50% per window evades it.
    A persisted first-computed baseline must defeat the slide."""

    def test_persisted_baseline_defeats_window_slide(self) -> None:
        # Only the tail of a slow drift is visible: self-window baseline is
        # mean(40, 42) = 41 -> 45 stays quiet. The persisted baseline (28.0,
        # frozen sweeps ago) must alarm.
        rows = [
            _row(900_000, avg_game_length=40.0),
            _row(950_000, avg_game_length=42.0),
            _row(1_000_000, avg_game_length=45.0),
        ]
        assert "game_length_drift" not in _alarm_names(rows)
        persisted = {"avg_game_length": 28.0, "source": "earliest_rows", "rows": 2}
        alarms = [
            a
            for a in mp.evaluate_watchdogs(rows, persisted_baseline=persisted)
            if a["watchdog"] == "game_length_drift"
        ]
        (alarm,) = alarms
        assert alarm["baseline_persisted"] is True
        assert alarm["baseline_avg_game_length"] == 28.0
        assert alarm["baseline_source"] == "earliest_rows"
        assert alarm["latest_avg_game_length"] == 45.0

    def test_persisted_baseline_wins_over_window_band(self) -> None:
        # First-computed wins over any window recompute — that is the point.
        rows = _baseline_rows() + [_row(200_000, avg_game_length=30.0)]
        persisted = {"avg_game_length": 10.0, "source": "band", "rows": 3}
        alarms = [
            a
            for a in mp.evaluate_watchdogs(rows, persisted_baseline=persisted)
            if a["watchdog"] == "game_length_drift"
        ]
        (alarm,) = alarms
        assert alarm["baseline_avg_game_length"] == 10.0
        assert alarm["baseline_persisted"] is True

    def test_persisted_baseline_arms_single_row_window(self) -> None:
        # A single length row cannot self-baseline (degrades to a warning),
        # but a persisted baseline keeps the dog fully armed.
        rows = [_row(600_000, avg_game_length=300.0)]
        persisted = {"avg_game_length": 28.0, "source": "band", "rows": 3}
        alarms = mp.evaluate_watchdogs(rows, persisted_baseline=persisted)
        names = {a["watchdog"] for a in alarms}
        assert "game_length_drift" in names
        assert "game_length_drift_no_baseline" not in names

    def test_unusable_persisted_falls_back_to_window(self) -> None:
        rows = _baseline_rows() + [_row(200_000, avg_game_length=56.0)]
        for bad in (None, "junk", {}, {"avg_game_length": math.nan},
                    {"avg_game_length": -5.0}, {"avg_game_length": True}):
            alarms = [
                a
                for a in mp.evaluate_watchdogs(rows, persisted_baseline=bad)
                if a["watchdog"] == "game_length_drift"
            ]
            (alarm,) = alarms
            assert alarm["baseline_persisted"] is False, bad
            assert alarm["baseline_avg_game_length"] == 28.0, bad
            assert alarm["baseline_source"] == "band", bad

    def test_persisted_baseline_healthy_latest_stays_quiet(self) -> None:
        rows = _baseline_rows() + [_row(200_000, avg_game_length=41.0)]
        persisted = {"avg_game_length": 28.0, "source": "band", "rows": 3}
        alarms = mp.evaluate_watchdogs(rows, persisted_baseline=persisted)
        assert "game_length_drift" not in {a["watchdog"] for a in alarms}


def _write_timeline(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _watchdog_cli(monkeypatch, capsys, *argv: str) -> list[dict]:
    monkeypatch.setattr(
        sys, "argv", ["milestone_probes.py", "watchdog", "--run-id", "test-run", *argv]
    )
    assert mp.main() == 0
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.strip()]


class TestBaselineFileCLI:
    """The watchdog subcommand persists the first-computed baseline to
    --baseline-file and reuses it on later sweeps; dry-run never writes;
    a corrupt file warns loudly and is left in place."""

    def test_first_sweep_persists_baseline(self, tmp_path, monkeypatch, capsys) -> None:
        timeline = tmp_path / "timeline.jsonl"
        _write_timeline(timeline, _baseline_rows() + [_row(200_000, avg_game_length=30.0)])
        baseline_file = tmp_path / "probes" / "test-run" / "drift-baseline.json"
        alarms = _watchdog_cli(
            monkeypatch, capsys, "--timeline", str(timeline),
            "--baseline-file", str(baseline_file),
        )
        assert alarms == []  # healthy sweep: stdout stays pure (empty) JSONL
        payload = json.loads(baseline_file.read_text(encoding="utf-8"))
        assert payload["schema_version"] == mp.BASELINE_SCHEMA_VERSION
        assert payload["run_id"] == "test-run"
        assert payload["avg_game_length"] == 28.0
        assert payload["source"] == "band"
        assert payload["rows"] == 3
        assert "computed_at_utc" in payload

    def test_second_sweep_reuses_persisted_across_window_slide(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        timeline = tmp_path / "timeline.jsonl"
        baseline_file = tmp_path / "drift-baseline.json"
        _write_timeline(timeline, _baseline_rows() + [_row(200_000, avg_game_length=30.0)])
        _watchdog_cli(
            monkeypatch, capsys, "--timeline", str(timeline),
            "--baseline-file", str(baseline_file),
        )
        first_content = baseline_file.read_text(encoding="utf-8")

        # the tail window has slid past the band: self-window would be quiet
        _write_timeline(
            timeline,
            [
                _row(900_000, avg_game_length=40.0),
                _row(950_000, avg_game_length=42.0),
                _row(1_000_000, avg_game_length=45.0),
            ],
        )
        alarms = _watchdog_cli(
            monkeypatch, capsys, "--timeline", str(timeline),
            "--baseline-file", str(baseline_file),
        )
        (alarm,) = [a for a in alarms if a["watchdog"] == "game_length_drift"]
        assert alarm["baseline_persisted"] is True
        assert alarm["baseline_avg_game_length"] == 28.0
        assert alarm["baseline_source"] == "band"
        assert alarm["run_id"] == "test-run"
        assert alarm["schema_version"] == mp.ALERT_SCHEMA_VERSION
        # written once — the second sweep must not touch the file
        assert baseline_file.read_text(encoding="utf-8") == first_content

    def test_no_persist_never_writes(self, tmp_path, monkeypatch, capsys) -> None:
        timeline = tmp_path / "timeline.jsonl"
        _write_timeline(timeline, _baseline_rows() + [_row(200_000, avg_game_length=30.0)])
        baseline_file = tmp_path / "drift-baseline.json"
        _watchdog_cli(
            monkeypatch, capsys, "--timeline", str(timeline),
            "--baseline-file", str(baseline_file), "--no-persist",
        )
        assert not baseline_file.exists()

    def test_corrupt_baseline_file_warns_and_is_left_alone(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        timeline = tmp_path / "timeline.jsonl"
        _write_timeline(timeline, _baseline_rows() + [_row(200_000, avg_game_length=56.0)])
        baseline_file = tmp_path / "drift-baseline.json"
        baseline_file.write_text("{not json", encoding="utf-8")
        alarms = _watchdog_cli(
            monkeypatch, capsys, "--timeline", str(timeline),
            "--baseline-file", str(baseline_file),
        )
        (warning,) = [a for a in alarms if a["watchdog"] == "drift_baseline_unreadable"]
        assert warning["severity"] == "warning"
        assert warning["baseline_file"] == str(baseline_file)
        # the dog still evaluated, from the window baseline
        (alarm,) = [a for a in alarms if a["watchdog"] == "game_length_drift"]
        assert alarm["baseline_persisted"] is False
        # never overwritten — a quiet rewrite would reset the baseline
        assert baseline_file.read_text(encoding="utf-8") == "{not json"

    def test_no_file_written_when_no_baseline_computable(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        timeline = tmp_path / "timeline.jsonl"
        _write_timeline(timeline, [_row(600_000, avg_game_length=28.0)])
        baseline_file = tmp_path / "drift-baseline.json"
        alarms = _watchdog_cli(
            monkeypatch, capsys, "--timeline", str(timeline),
            "--baseline-file", str(baseline_file),
        )
        assert not baseline_file.exists()
        assert "game_length_drift_no_baseline" in {a["watchdog"] for a in alarms}


class TestStrengthRegression:
    def test_no_alarm_at_exact_ten_point_drop(self) -> None:
        # boundary unified with the other dogs: exactly -10 points is quiet
        rows = _baseline_rows() + [
            _row(200_000, max_damage=0.80),
            _row(300_000, max_damage=0.70),
        ]
        assert "strength_regression" not in _alarm_names(rows)

    def test_exact_drop_is_float_representation_proof(self) -> None:
        # 0.85-0.75 and 0.80-0.70 land on opposite sides of 0.10 in binary
        # floats; the epsilon must make them behave identically (both quiet)
        rows = _baseline_rows() + [
            _row(200_000, max_damage=0.85),
            _row(300_000, max_damage=0.75),
        ]
        assert "strength_regression" not in _alarm_names(rows)

    def test_alarm_strictly_beyond_ten_points(self) -> None:
        for peak, latest in [(0.80, 0.69), (0.85, 0.74)]:
            rows = _baseline_rows() + [
                _row(200_000, max_damage=peak),
                _row(300_000, max_damage=latest),
            ]
            assert "strength_regression" in _alarm_names(rows), (peak, latest)

    def test_no_alarm_below_threshold(self) -> None:
        rows = _baseline_rows() + [
            _row(200_000, max_damage=0.80),
            _row(300_000, max_damage=0.71),  # -9 points
        ]
        assert "strength_regression" not in _alarm_names(rows)

    def test_peak_is_run_lifetime_peak(self) -> None:
        # regression is judged against the run's own peak, not the previous row
        rows = _baseline_rows() + [
            _row(200_000, max_damage=0.80),
            _row(250_000, max_damage=0.76),
            _row(300_000, max_damage=0.69),
        ]
        (alarm,) = _alarms_of(rows, "strength_regression")
        assert alarm["peak_max_damage_win_rate"] == 0.80
        assert alarm["peak_milestone_games"] == 200_000
        assert alarm["latest_max_damage_win_rate"] == 0.69
        assert alarm["severity"] == "alarm"

    def test_fidelity_matched(self) -> None:
        # a high-fi peak must not be held against a low-fi latest read
        rows = [
            _row(100_000, fidelity="high", max_damage=0.85),
            _row(150_000, max_damage=0.60),
            _row(200_000, max_damage=0.58),
        ]
        assert "strength_regression" not in _alarm_names(rows)

    def test_single_row_no_alarm(self) -> None:
        assert "strength_regression" not in _alarm_names([_row(100_000, max_damage=0.10)])

    def test_improving_run_no_alarm(self) -> None:
        rows = _baseline_rows() + [_row(200_000, max_damage=0.75)]
        assert "strength_regression" not in _alarm_names(rows)


class TestStrengthRegressionPerFidelity:
    """L1 of the #500 review: the dog must evaluate the latest row of EACH
    fidelity — a newer healthy high-fi row must not mask a low-fi collapse."""

    def test_newer_high_fi_row_does_not_mask_low_fi_collapse(self) -> None:
        rows = [
            _row(150_000, max_damage=0.62),
            _row(200_000, max_damage=0.45),
            _row(200_000, fidelity="high", max_damage=0.80, completed=210_000),
        ]
        (alarm,) = _alarms_of(rows, "strength_regression")
        assert alarm["fidelity"] == "low"
        assert alarm["latest_max_damage_win_rate"] == 0.45
        assert alarm["peak_max_damage_win_rate"] == 0.62

    def test_alarm_per_collapsed_fidelity(self) -> None:
        rows = [
            _row(100_000, max_damage=0.60),
            _row(100_000, fidelity="high", max_damage=0.70, completed=101_000),
            _row(200_000, max_damage=0.40),
            _row(200_000, fidelity="high", max_damage=0.50, completed=201_000),
        ]
        alarms = _alarms_of(rows, "strength_regression")
        assert {alarm["fidelity"] for alarm in alarms} == {"low", "high"}

    def test_healthy_fidelity_quiet_alongside_collapsed_one(self) -> None:
        rows = [
            _row(100_000, max_damage=0.60),
            _row(100_000, fidelity="high", max_damage=0.70, completed=101_000),
            _row(200_000, max_damage=0.40),
            _row(200_000, fidelity="high", max_damage=0.68, completed=201_000),
        ]
        (alarm,) = _alarms_of(rows, "strength_regression")
        assert alarm["fidelity"] == "low"

    def test_single_row_per_fidelity_stays_quiet(self) -> None:
        rows = [
            _row(100_000, max_damage=0.60),
            _row(200_000, fidelity="high", max_damage=0.20, completed=201_000),
        ]
        assert "strength_regression" not in _alarm_names(rows)


class TestPolicyEntropyFloor:
    def test_alarm_below_floor(self) -> None:
        rows = _baseline_rows() + [_row(200_000, avg_game_length=28.0, policy_entropy=0.34)]
        assert "policy_entropy_floor" in _alarm_names(rows)

    def test_no_alarm_at_floor(self) -> None:
        rows = _baseline_rows() + [_row(200_000, avg_game_length=28.0, policy_entropy=0.35)]
        assert "policy_entropy_floor" not in _alarm_names(rows)

    def test_uses_latest_entropy(self) -> None:
        # an early dip that recovered must not alarm
        rows = _baseline_rows() + [
            _row(150_000, avg_game_length=28.0, policy_entropy=0.20),
            _row(200_000, avg_game_length=28.0, policy_entropy=0.90),
        ]
        assert "policy_entropy_floor" not in _alarm_names(rows)


class TestDataQuality:
    """M2: NaN/Infinity watched values must be quarantined (never silently
    disable a dog) and surfaced as a warning; rows without completed_games
    sort last, never first."""

    def test_nan_latest_length_does_not_mask_prior_spike(self) -> None:
        rows = _baseline_rows() + [
            _row(200_000, avg_game_length=90.0),
            _row(300_000, avg_game_length=math.nan),
        ]
        assert "game_length_drift" in _alarm_names(rows)
        assert "timeline_data_quality" in _alarm_names(rows)

    def test_nan_in_baseline_band_excluded_from_mean(self) -> None:
        rows = [
            _row(30_000, avg_game_length=27.0),
            _row(60_000, avg_game_length=math.nan),
            _row(100_000, avg_game_length=29.0),
            _row(200_000, avg_game_length=56.0),
        ]
        (alarm,) = _alarms_of(rows, "game_length_drift")
        assert alarm["baseline_avg_game_length"] == 28.0  # mean of 27, 29
        assert "timeline_data_quality" in _alarm_names(rows)

    def test_nan_entropy_falls_back_to_last_finite_value(self) -> None:
        rows = _baseline_rows() + [
            _row(150_000, avg_game_length=28.0, policy_entropy=0.20),
            _row(200_000, avg_game_length=28.0, policy_entropy=math.nan),
        ]
        # latest FINITE entropy is 0.20 -> alarm; the NaN row cannot mute it
        assert "policy_entropy_floor" in _alarm_names(rows)

    def test_nan_win_rate_quarantined(self) -> None:
        rows = _baseline_rows() + [
            _row(200_000, max_damage=0.80),
            _row(300_000, max_damage=math.nan),
        ]
        # the NaN row is excluded; latest finite read (0.80) is the peak itself
        assert "strength_regression" not in _alarm_names(rows)
        assert "timeline_data_quality" in _alarm_names(rows)

    def test_infinity_treated_like_nan(self) -> None:
        rows = _baseline_rows() + [_row(200_000, avg_game_length=math.inf)]
        assert "game_length_drift" not in _alarm_names(rows)
        (warning,) = _alarms_of(rows, "timeline_data_quality")
        assert warning["severity"] == "warning"
        assert warning["non_finite_rows"] == 1

    def test_data_quality_counts_rows(self) -> None:
        rows = _baseline_rows() + [
            _row(200_000, avg_game_length=math.nan, policy_entropy=math.nan),
            _row(300_000, max_damage=math.nan),
        ]
        (warning,) = _alarms_of(rows, "timeline_data_quality")
        assert warning["non_finite_rows"] == 2

    def test_missing_completed_games_sorts_last_not_first(self) -> None:
        # the malformed row is FIRST in file order; if it sorted first the
        # healthy old rows would stay "latest" and its 0.10 entropy would be
        # invisible — sorted last it becomes the latest read and must alarm
        malformed = _row(
            999_000, avg_game_length=28.0, policy_entropy=0.10, completed=_NO_COMPLETED
        )
        rows = [malformed] + _baseline_rows()
        assert "policy_entropy_floor" in _alarm_names(rows)

    def test_nan_completed_games_sorts_last(self) -> None:
        malformed = _row(
            999_000, avg_game_length=28.0, policy_entropy=0.10, completed=math.nan
        )
        rows = [malformed] + _baseline_rows()
        assert "policy_entropy_floor" in _alarm_names(rows)
        assert "timeline_data_quality" in _alarm_names(rows)

    def test_healthy_timeline_has_no_data_quality_warning(self) -> None:
        assert "timeline_data_quality" not in _alarm_names(_baseline_rows())

    def test_non_finite_milestone_games_on_latest_row_warns(self) -> None:
        # Residual 2 of the #500 verify review: a NaN milestone_games on the
        # latest row used to fall into the quiet "establishing the band"
        # branch with ZERO output — it must be quarantined with a warning.
        for bad in (math.nan, math.inf):
            rows = _baseline_rows() + [
                _row(bad, avg_game_length=300.0, completed=300_000),
            ]
            (warning,) = _alarms_of(rows, "timeline_data_quality")
            assert warning["non_finite_rows"] == 1, bad

    def test_nan_milestone_on_mid_row_does_not_poison_band(self) -> None:
        rows = _baseline_rows() + [
            _row(math.nan, avg_game_length=28.0, completed=150_000),
            _row(200_000, avg_game_length=56.0),
        ]
        (alarm,) = _alarms_of(rows, "game_length_drift")
        assert alarm["baseline_avg_game_length"] == 28.0  # band mean untouched
        assert "timeline_data_quality" in _alarm_names(rows)

    def test_string_milestone_games_warns_without_crashing(self) -> None:
        # a string milestone_games used to raise TypeError in the band
        # comparison; now it reads as missing and the row is warned about
        rows = [
            _row("60k", avg_game_length=27.0, completed=30_000),
            _row(60_000, avg_game_length=28.0),
            _row(100_000, avg_game_length=29.0),
            _row(200_000, avg_game_length=56.0),
        ]
        (alarm,) = _alarms_of(rows, "game_length_drift")
        assert alarm["baseline_avg_game_length"] == 28.5  # mean of 28, 29
        assert "timeline_data_quality" in _alarm_names(rows)


class TestRobustness:
    def test_empty_timeline(self) -> None:
        assert mp.evaluate_watchdogs([]) == []

    def test_rows_without_collapse_or_metrics(self) -> None:
        rows = [
            {"fidelity": "low", "milestone_games": 50_000, "completed_games": 50_000},
            {"fidelity": "high", "milestone_games": 50_000, "metrics": {}},
        ]
        assert mp.evaluate_watchdogs(rows) == []

    def test_unsorted_rows_are_sorted_by_completed_games(self) -> None:
        healthy_last = _row(200_000, avg_game_length=29.0)
        spike = _row(150_000, avg_game_length=90.0)
        rows = [healthy_last, spike] + _baseline_rows()
        assert "game_length_drift" not in _alarm_names(rows)

    def test_multiple_alarms_fire_together(self) -> None:
        rows = _baseline_rows() + [
            _row(200_000, max_damage=0.80),
            _row(300_000, avg_game_length=70.0, policy_entropy=0.10, max_damage=0.40),
        ]
        assert _alarm_names(rows) == {
            "game_length_drift",
            "strength_regression",
            "policy_entropy_floor",
        }


class TestPendingMilestones:
    @staticmethod
    def _status(iterations, gpi: int = 1600, started_from: int = 0) -> dict:
        if isinstance(iterations, int):
            iterations = range(1, iterations + 1)
        return {
            "run_id": "foundation-test",
            "status": "running",
            "games_per_iteration": gpi,
            "started_from_completed_games": started_from,
            "latest_checkpoint_path": "/x/run/iteration-9999/transformer-policy.pt",
            "completed_iterations": [
                {
                    "iteration": i,
                    "checkpoint_path": f"/x/run/iteration-{i:04d}/transformer-policy.pt",
                }
                for i in iterations
            ],
        }

    def test_maps_milestones_to_nearest_iteration(self) -> None:
        plan = mp.pending_milestones(self._status(286), set(), step=100_000)
        assert plan["completed_games"] == 457_600
        milestones = {p["milestone_games"]: p["iteration"] for p in plan["pending"]}
        # 100k/1600 = 62.5 -> iteration 62 or 63 are equidistant; 200k -> 125
        assert milestones[100_000] in (62, 63)
        assert milestones[200_000] == 125
        assert milestones[300_000] in (187, 188)
        assert milestones[400_000] == 250
        assert 500_000 not in milestones  # not reached yet
        assert all(p["action"] == "probe" for p in plan["pending"])
        assert all(p["distance_games"] <= 800 for p in plan["pending"])

    def test_ledger_filtering_is_idempotent(self) -> None:
        plan = mp.pending_milestones(self._status(286), {100_000, 200_000}, step=100_000)
        assert [p["milestone_games"] for p in plan["pending"]] == [300_000, 400_000]

    def test_local_name_convention(self) -> None:
        plan = mp.pending_milestones(self._status(286), set(), step=100_000)
        item = plan["pending"][1]
        assert item["local_name"] == f"foundation-test-i{item['iteration']}.pt"
        assert item["remote_checkpoint"].endswith(
            f"iteration-{item['iteration']:04d}/transformer-policy.pt"
        )

    def test_young_run_has_no_pending(self) -> None:
        plan = mp.pending_milestones(self._status(30), set(), step=100_000)
        assert plan["completed_games"] == 48_000
        assert plan["pending"] == []

    def test_continuation_skips_prior_run_milestones(self) -> None:
        # continuation from 200k: milestones <= started_from belong to the parent
        plan = mp.pending_milestones(
            self._status(100, started_from=200_000), set(), step=100_000
        )
        assert plan["completed_games"] == 360_000
        assert [p["milestone_games"] for p in plan["pending"]] == [300_000]


class TestMilestoneDistanceGuard:
    """M4: rotated/retained-away checkpoints must not be silently probed as a
    far-away milestone — beyond max-distance the item comes back action=skip."""

    def test_rotated_early_checkpoints_skip_far_milestones(self) -> None:
        # iterations 82..286 only (early ones rotated away): milestone 100k's
        # nearest checkpoint is 31,200 games away -> skip, 200k+ still probe
        plan = mp.pending_milestones(
            TestPendingMilestones._status(range(82, 287)), set(), step=100_000
        )
        by_milestone = {p["milestone_games"]: p for p in plan["pending"]}
        skip = by_milestone[100_000]
        assert skip["action"] == "skip"
        assert skip["distance_games"] == 31_200
        assert "nearest checkpoint" in skip["skip_reason"]
        assert by_milestone[200_000]["action"] == "probe"
        assert by_milestone[400_000]["action"] == "probe"

    def test_distance_at_exact_max_still_probes(self) -> None:
        # boundary: skip only strictly beyond max_distance
        plan = mp.pending_milestones(
            TestPendingMilestones._status(range(82, 287)),
            set(),
            step=100_000,
            max_distance=31_200,
        )
        by_milestone = {p["milestone_games"]: p for p in plan["pending"]}
        assert by_milestone[100_000]["action"] == "probe"
        assert by_milestone[100_000]["distance_games"] == 31_200

    def test_normal_run_never_skips(self) -> None:
        plan = mp.pending_milestones(TestPendingMilestones._status(286), set(), step=100_000)
        assert all(p["action"] == "probe" for p in plan["pending"])
        assert all(p["skip_reason"] is None for p in plan["pending"])


class TestResolvePools:
    """Eval-pool resolution for the sweep's cross-pool Pearson read:
    POKEZERO_POOL_SELF / POKEZERO_POOL_FP override the frozen-v2 defaults,
    labels key ledger entries and pearson filenames, and explicitly selecting
    a retired v1 pool carries a deprecation warning (v1 pools store v1-encoded
    observations that the v2 schema guards refuse — they cannot score v2
    (obsv2) checkpoints)."""

    _REPO = Path("/repo")

    def test_defaults_are_the_frozen_v2_pools(self) -> None:
        # the default flip (review #509 M2): pinned so a silent revert to the
        # v1 pools — which cannot score v2 checkpoints — fails the suite
        assert mp.DEFAULT_POOL_SELF == "runs/pool-self-v2-20260705/pool-self-v2.jsonl"
        assert mp.DEFAULT_POOL_FP == "runs/pool-fp-v2-20260705/pool-fp-v2.jsonl"

    def test_defaults_resolve_under_repo_root(self) -> None:
        pools = mp.resolve_pools(self._REPO, {})
        assert pools["self"]["path"] == str(self._REPO / mp.DEFAULT_POOL_SELF)
        assert pools["self"]["source"] == "default"
        assert pools["fp"]["path"] == str(self._REPO / mp.DEFAULT_POOL_FP)
        assert pools["fp"]["source"] == "default"

    def test_default_labels_derive_from_the_v2_filename_stems(self) -> None:
        # labels come from the actual filename stem, NOT from a mapping keyed
        # off the defaults — review #509 M2: the flipped defaults must never
        # write v2 results under the historical v1 ledger keys
        pools = mp.resolve_pools(self._REPO, {})
        assert pools["self"]["label"] == "pool-self-v2"
        assert pools["fp"]["label"] == "pool-fp-v2"

    def test_v2_defaults_do_not_warn(self) -> None:
        # review #509 M2: the pre-flip code blared false v1 deprecation
        # banners after the documented DEFAULT_POOL_* flip
        assert mp.resolve_pools(self._REPO, {})["warnings"] == []

    def test_explicit_v1_pools_warn_and_keep_historical_labels(self) -> None:
        # v1-era ledgers/pearson files were written under these labels; an
        # explicitly selected v1 pool must keep them AND warn that v1 pools
        # cannot score v2 checkpoints (the only remaining v1 path is via env)
        env = {mp.POOL_SELF_ENV: mp.V1_POOL_SELF, mp.POOL_FP_ENV: mp.V1_POOL_FP}
        pools = mp.resolve_pools(self._REPO, env)
        assert pools["self"]["label"] == "pool-self-v1"
        assert pools["fp"]["label"] == "pool-fp-v1"
        assert len(pools["warnings"]) == 2
        for warning, env_var in zip(
            pools["warnings"], (mp.POOL_SELF_ENV, mp.POOL_FP_ENV)
        ):
            assert "obsv2" in warning
            assert env_var in warning

    def test_v1_detection_keys_off_the_v1_paths_not_the_defaults(self) -> None:
        # regression for review #509 M2: "is v1" was implemented as "is the
        # default", so ANY future default flip would silently relabel the new
        # pools as v1. Simulate the next flip and assert both halves.
        original = mp.DEFAULT_POOL_SELF
        mp.DEFAULT_POOL_SELF = "runs/pool-self-v3-20270101/pool-self-v3.jsonl"
        try:
            pools = mp.resolve_pools(self._REPO, {})
            assert pools["self"]["label"] == "pool-self-v3"
            assert pools["warnings"] == []
            pools = mp.resolve_pools(self._REPO, {mp.POOL_SELF_ENV: mp.V1_POOL_SELF})
            assert pools["self"]["label"] == "pool-self-v1"
            assert len(pools["warnings"]) == 1
        finally:
            mp.DEFAULT_POOL_SELF = original

    def test_env_overrides_take_precedence_without_warnings(self) -> None:
        env = {
            mp.POOL_SELF_ENV: "/pools/pool-self-v2-20260705/pool-self-v2.jsonl",
            mp.POOL_FP_ENV: "/pools/pool-fp-v2-20260705/pool-fp-v2.jsonl",
        }
        pools = mp.resolve_pools(self._REPO, env)
        assert pools["self"] == {
            "path": "/pools/pool-self-v2-20260705/pool-self-v2.jsonl",
            "label": "pool-self-v2",
            "source": "env",
        }
        assert pools["fp"] == {
            "path": "/pools/pool-fp-v2-20260705/pool-fp-v2.jsonl",
            "label": "pool-fp-v2",
            "source": "env",
        }
        assert pools["warnings"] == []

    def test_partial_v1_override_warns_only_about_that_pool(self) -> None:
        env = {mp.POOL_FP_ENV: mp.V1_POOL_FP}
        pools = mp.resolve_pools(self._REPO, env)
        assert pools["self"]["label"] == "pool-self-v2"  # v2 default, quiet
        (warning,) = pools["warnings"]
        assert mp.POOL_FP_ENV in warning
        assert "pool-fp-v1" in warning

    def test_relative_override_resolves_under_repo_root(self) -> None:
        env = {mp.POOL_SELF_ENV: "runs/pool-self-v2-20260705/pool-self-v2.jsonl"}
        pools = mp.resolve_pools(self._REPO, env)
        assert pools["self"]["path"] == str(
            self._REPO / "runs/pool-self-v2-20260705/pool-self-v2.jsonl"
        )
        assert pools["self"]["source"] == "env"

    def test_tilde_override_expands_to_home(self) -> None:
        env = {mp.POOL_FP_ENV: "~/pools/pool-fp-v2b.jsonl"}
        pools = mp.resolve_pools(self._REPO, env)
        assert pools["fp"]["path"] == os.path.realpath(
            os.path.expanduser("~/pools/pool-fp-v2b.jsonl")
        )

    def test_empty_env_value_means_unset(self) -> None:
        # matches the shell's ${VAR:-default}: an exported empty string must
        # fall back to the default, never yield an empty pool path
        pools = mp.resolve_pools(self._REPO, {mp.POOL_SELF_ENV: "", mp.POOL_FP_ENV: ""})
        assert pools["self"]["source"] == "default"
        assert pools["fp"]["source"] == "default"
        assert pools["warnings"] == []

    def test_override_spelling_the_v1_path_keeps_v1_label_and_warning(self) -> None:
        # an env var pointing at the v1 pool (any spelling) is still a v1
        # pool: historical label and deprecation warning, but source stays env
        env = {mp.POOL_SELF_ENV: str(self._REPO / mp.V1_POOL_SELF)}
        pools = mp.resolve_pools(self._REPO, env)
        assert pools["self"]["label"] == "pool-self-v1"
        assert pools["self"]["source"] == "env"
        assert len(pools["warnings"]) == 1

    def test_dotdot_spelling_cannot_evade_v1_detection(self) -> None:
        # review #509 L1: lexical Path equality let a ..-spelled v1 path
        # dodge the deprecation warning AND fork a fresh ledger label for
        # the same pool file; paths normalize (realpath) before comparison
        env = {mp.POOL_SELF_ENV: str(self._REPO / "runs" / ".." / mp.V1_POOL_SELF)}
        pools = mp.resolve_pools(self._REPO, env)
        assert pools["self"]["label"] == "pool-self-v1"
        assert pools["self"]["path"] == str(
            Path(os.path.realpath(self._REPO / mp.V1_POOL_SELF))
        )
        assert len(pools["warnings"]) == 1

    def test_symlink_spelling_cannot_evade_v1_detection(self, tmp_path) -> None:
        # review #509 L1, symlink flavor
        repo = tmp_path / "repo"
        v1 = repo / mp.V1_POOL_SELF
        v1.parent.mkdir(parents=True)
        v1.write_text("{}\n", encoding="utf-8")
        alias = tmp_path / "alias.jsonl"
        alias.symlink_to(v1)
        pools = mp.resolve_pools(repo, {mp.POOL_SELF_ENV: str(alias)})
        assert pools["self"]["label"] == "pool-self-v1"
        assert len(pools["warnings"]) == 1

    def test_label_is_sanitized_filename_stem(self) -> None:
        env = {mp.POOL_SELF_ENV: "/pools/pool self (v2)!.jsonl"}
        pools = mp.resolve_pools(self._REPO, env)
        assert pools["self"]["label"] == "pool-self-v2"

    @staticmethod
    def _fatal_message(env: dict) -> str:
        # stdlib-only stand-in for pytest.raises: the suite runs under
        # `python -m unittest discover`, so this file must import cleanly
        # without pytest installed
        try:
            mp.resolve_pools(TestResolvePools._REPO, env)
        except SystemExit as exc:
            return str(exc)
        raise AssertionError("expected resolve_pools to raise SystemExit")

    def test_colliding_labels_are_fatal(self) -> None:
        # both entry["pools"] ledger keys and pearson-<label>-<milestone>.json
        # filenames would collide — refuse instead of silently overwriting
        env = {
            mp.POOL_SELF_ENV: "/a/pool-v2.jsonl",
            mp.POOL_FP_ENV: "/b/pool-v2.jsonl",
        }
        assert "same label" in self._fatal_message(env)

    def test_underivable_label_is_fatal(self) -> None:
        message = self._fatal_message({mp.POOL_SELF_ENV: "/pools/___.jsonl"})
        assert "pool label" in message


class TestPoolsCLITransport:
    """The `pools` CLI transports resolution to the bash sweep as
    NUL-delimited key/value pairs (review #509 L2 — a TSV row mis-splits on
    tab/newline-containing paths; NUL cannot appear in paths or env values),
    and --hash pins each existing pool file with a streaming sha256, computed
    once per sweep (review #509 M1)."""

    @staticmethod
    def _pairs(monkeypatch, capsys, repo: Path, *, with_hash: bool) -> list[tuple[str, str]]:
        monkeypatch.setattr(
            sys,
            "argv",
            ["milestone_probes.py", "pools", "--repo", str(repo), "--format", "nul"]
            + (["--hash"] if with_hash else []),
        )
        assert mp.main() == 0
        out = capsys.readouterr().out
        parts = out.split("\0")
        assert parts[-1] == ""  # every value is NUL-terminated
        parts = parts[:-1]
        assert len(parts) % 2 == 0
        return list(zip(parts[0::2], parts[1::2]))

    def test_nul_transport_survives_tab_and_newline_paths(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        evil = tmp_path / "pool\tself\nv9.jsonl"
        monkeypatch.setenv(mp.POOL_SELF_ENV, str(evil))
        monkeypatch.delenv(mp.POOL_FP_ENV, raising=False)
        pairs = dict(self._pairs(monkeypatch, capsys, tmp_path, with_hash=False))
        # the path round-trips byte-exact — a TSV hop would have split it
        assert pairs["self.path"] == os.path.realpath(str(evil))
        assert pairs["self.label"] == "pool-self-v9"
        assert pairs["self.source"] == "env"
        assert pairs["fp.source"] == "default"
        assert "self.sha256" not in pairs  # no --hash (dry-run)

    def test_hash_pins_the_exact_pool_bytes(self, tmp_path, monkeypatch, capsys) -> None:
        content = b'{"observation": [1, 2, 3]}\n' * 257
        pool = tmp_path / mp.DEFAULT_POOL_SELF
        pool.parent.mkdir(parents=True)
        pool.write_bytes(content)
        monkeypatch.delenv(mp.POOL_SELF_ENV, raising=False)
        monkeypatch.delenv(mp.POOL_FP_ENV, raising=False)
        pairs = dict(self._pairs(monkeypatch, capsys, tmp_path, with_hash=True))
        assert pairs["self.sha256"] == hashlib.sha256(content).hexdigest()
        # absent pool (the fp default is not on disk here) hashes to "" —
        # existence stays a preflight concern, resolution never dies on it
        assert pairs["fp.sha256"] == ""

    def test_warnings_travel_as_warning_pairs(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv(mp.POOL_SELF_ENV, mp.V1_POOL_SELF)
        monkeypatch.delenv(mp.POOL_FP_ENV, raising=False)
        pairs = self._pairs(monkeypatch, capsys, tmp_path, with_hash=False)
        banner = "\n".join(value for key, value in pairs if key == "warning")
        assert "pool-self-v1" in banner
        assert mp.POOL_SELF_ENV in banner


class TestPearsonPoolPin:
    """Review #509 M1: every pearson artifact payload and every ledger line
    pins the exact pool file that produced its numbers (absolute path +
    sha256). annotate-pearson injects the pin; record embeds it and refuses
    unpinned or misattributed payloads — a label alone does not identify a
    pool file."""

    _SHA = "0123456789abcdef" * 4

    def _annotate(self, path: Path, **overrides) -> int:
        args = {
            "json": str(path),
            "label": "pool-self-v2",
            "pool_path": "/pools/pool-self-v2-20260705/pool-self-v2.jsonl",
            "pool_sha256": self._SHA,
        }
        args.update(overrides)
        return mp._cmd_annotate_pearson(argparse.Namespace(**args))

    def _record(self, ledger: Path, pearson_specs: list[str]) -> int:
        return mp._cmd_record(
            argparse.Namespace(
                ledger=str(ledger),
                run_id="test-run",
                milestone=100_000,
                iteration=7,
                checkpoint="checkpoints/curated/test-run-i7.pt",
                games_at=98_304,
                pearson=pearson_specs,
                hazard=None,
                hazard_label=None,
                skip_reason=None,
            )
        )

    def test_annotate_injects_the_pool_pin(self, tmp_path) -> None:
        artifact = tmp_path / "pearson.json.tmp"
        artifact.write_text(
            json.dumps({"pearson": 0.91, "n_states": 4096}), encoding="utf-8"
        )
        assert self._annotate(artifact) == 0
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert payload["pearson"] == 0.91  # value-calibration fields intact
        assert payload["n_states"] == 4096
        assert payload["pool"] == {
            "label": "pool-self-v2",
            "path": "/pools/pool-self-v2-20260705/pool-self-v2.jsonl",
            "sha256": self._SHA,
        }

    def test_annotate_refuses_empty_pin_fields(self, tmp_path) -> None:
        artifact = tmp_path / "pearson.json.tmp"
        original = json.dumps({"pearson": 0.91})
        artifact.write_text(original, encoding="utf-8")
        for overrides in ({"pool_path": ""}, {"pool_sha256": ""}):
            try:
                self._annotate(artifact, **overrides)
            except SystemExit as exc:
                assert "pool" in str(exc)
            else:
                raise AssertionError("expected annotate-pearson to raise SystemExit")
        # refused BEFORE writing: the artifact is untouched
        assert artifact.read_text(encoding="utf-8") == original

    def test_record_embeds_the_pool_pin_in_the_ledger(self, tmp_path, capsys) -> None:
        artifact = tmp_path / "pearson-pool-self-v2-01234567-100000.json"
        artifact.write_text(json.dumps({"pearson": 0.91}), encoding="utf-8")
        self._annotate(artifact)
        ledger = tmp_path / "ledger.jsonl"
        assert self._record(ledger, [f"pool-self-v2={artifact}"]) == 0
        capsys.readouterr()
        (line,) = ledger.read_text(encoding="utf-8").splitlines()
        row = json.loads(line)["pools"]["pool-self-v2"]
        assert row["pearson"] == 0.91
        assert row["pool"]["path"] == "/pools/pool-self-v2-20260705/pool-self-v2.jsonl"
        assert row["pool"]["sha256"] == self._SHA
        assert row["pearson_file"] == str(artifact)

    def test_record_refuses_unpinned_payloads(self, tmp_path, capsys) -> None:
        # a pre-#509 artifact (no pool pin) must never produce a ledger line
        # whose numbers cannot be attributed to a specific pool file
        artifact = tmp_path / "pearson-pool-self-v2-100000.json"
        artifact.write_text(json.dumps({"pearson": 0.91}), encoding="utf-8")
        ledger = tmp_path / "ledger.jsonl"
        try:
            self._record(ledger, [f"pool-self-v2={artifact}"])
        except SystemExit as exc:
            assert "pool pin" in str(exc)
        else:
            raise AssertionError("expected record to raise SystemExit")
        assert not ledger.exists()

    def test_record_refuses_misattributed_payloads(self, tmp_path, capsys) -> None:
        artifact = tmp_path / "pearson.json"
        artifact.write_text(json.dumps({"pearson": 0.91}), encoding="utf-8")
        self._annotate(artifact)  # pinned to pool-self-v2
        ledger = tmp_path / "ledger.jsonl"
        try:
            self._record(ledger, [f"pool-fp-v2={artifact}"])
        except SystemExit as exc:
            assert "misattribute" in str(exc)
        else:
            raise AssertionError("expected record to raise SystemExit")
        assert not ledger.exists()


_SWEEP_SH = _SCRIPT.with_name("milestone_probes.sh")

# Distinctive, non-round sleep duration: the liveness test necessarily leaves
# one orphaned `sleep` behind (that is the mechanism under test); teardown
# pkills exactly this command line, and a missed pkill self-cleans in ~1 min.
_ORPHAN_SECS = "63.79"

# Reproduces the sweep's lock lifecycle around one run_with_timeout call:
# fd 9 opened on the lockfile and flocked by the python helper (exactly the
# sweep preamble), then the SHIPPED run_with_timeout — extracted from
# milestone_probes.sh, not a copy — runs one command with a PATH that hides
# coreutils `timeout`, forcing the background-killer fallback used on stock
# macOS. When this shell exits, fd 9 closes; only a process that leaked the
# fd can still hold the flock.
_LOCK_DRIVER = """\
set -euo pipefail
repo="$1"; lock="$2"; stubbin="$3"; secs="$4"; shift 4
exec 9>"$lock"
python3 "$repo/scripts/milestone_probes.py" lock --fd 9
fn="$(sed -n '/^run_with_timeout()/,/^}/p' "$repo/scripts/milestone_probes.sh")"
[ -n "$fn" ] || { echo "could not extract run_with_timeout" >&2; exit 90; }
eval "$fn"
ln -s "$(command -v sleep)" "$stubbin/sleep"
if PATH="$stubbin" command -v timeout >/dev/null 2>&1; then
  echo "timeout unexpectedly on the stub PATH" >&2; exit 91
fi
rc=0
PATH="$stubbin" run_with_timeout "$secs" "$@" || rc=$?
echo "run_with_timeout_rc=$rc"
"""


class TestSweepLockLiveness:
    """The no-coreutils-timeout fallback in run_with_timeout (stock macOS)
    must not leak the sweep flock: `kill "$killer_pid"` reaps the killer
    subshell but not its already-forked sleep, and an orphaned sleep that
    inherited fd 9 held runs/milestone-probes/.sweep.lock for up to
    POKEZERO_CP_TIMEOUT after the sweep exited — every cron sweep in that
    window bailed with "another sweep holds the lock"."""

    def _drive(self, tmp_path: Path, secs: str, *cmd: str) -> tuple:
        driver = tmp_path / "driver.sh"
        driver.write_text(_LOCK_DRIVER, encoding="utf-8")
        lock = tmp_path / ".sweep.lock"
        stubbin = tmp_path / "stubbin"
        stubbin.mkdir()
        # Output goes to FILES, not pipes: an fd-leaking killer subshell
        # orphans a sleep that would hold pipe write-ends open long after
        # bash exits, and capture_output would block on EOF until the orphan
        # died — masking the crisp flock assertion below with a timeout.
        out_path = tmp_path / "driver.out"
        err_path = tmp_path / "driver.err"
        with out_path.open("w") as out, err_path.open("w") as err:
            proc = subprocess.run(
                ["bash", str(driver), str(_SWEEP_SH.parents[1]), str(lock), str(stubbin), secs, *cmd],
                stdout=out, stderr=err, timeout=30,
            )
        stdout = out_path.read_text(encoding="utf-8")
        stderr = err_path.read_text(encoding="utf-8")
        return proc.returncode, stdout, stderr, lock

    def _assert_lock_free(self, lock: Path) -> None:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise AssertionError(
                    "sweep lock still held after the sweep shell exited — "
                    "run_with_timeout leaked fd 9 to an orphaned sleep"
                )
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _reap_orphan(self) -> None:
        subprocess.run(
            ["pkill", "-x", "-f", f"sleep {_ORPHAN_SECS}"],
            capture_output=True, check=False,
        )

    def test_lock_free_after_command_finishes_before_timeout(self, tmp_path) -> None:
        # The common sweep case: kubectl cp finishes well inside its timeout.
        # The 1s command guarantees the killer subshell has forked its sleep
        # by the time run_with_timeout kills the subshell — the exact moment
        # the buggy version orphaned a lock-holding sleep.
        try:
            rc, stdout, stderr, lock = self._drive(tmp_path, _ORPHAN_SECS, "sleep", "1")
            assert rc == 0, stderr
            assert "run_with_timeout_rc=0" in stdout
            self._assert_lock_free(lock)
        finally:
            self._reap_orphan()

    def test_fallback_still_kills_a_hung_command(self, tmp_path) -> None:
        # Guard the other direction: the fd hygiene must not break the
        # killer itself. A hung command dies at the timeout (rc > 128) and
        # the lock is free immediately afterwards.
        try:
            rc, stdout, stderr, lock = self._drive(tmp_path, "1", "sleep", _ORPHAN_SECS)
            assert rc == 0, stderr
            rc_lines = [l for l in stdout.splitlines() if l.startswith("run_with_timeout_rc=")]
            assert len(rc_lines) == 1, stdout
            assert int(rc_lines[0].partition("=")[2]) > 128
            self._assert_lock_free(lock)
        finally:
            self._reap_orphan()
