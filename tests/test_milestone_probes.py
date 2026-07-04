"""Unit tests for the ecology-watchdog and milestone-planning logic in
scripts/milestone_probes.py (WS-3 items 2-3, docs/next_train_readiness_plan.md).

The module is stdlib-only and lives outside the package, so it is loaded
straight from the scripts directory.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "milestone_probes.py"
_spec = importlib.util.spec_from_file_location("milestone_probes", _SCRIPT)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def _row(
    milestone: int,
    *,
    fidelity: str = "low",
    avg_game_length: float | None = None,
    policy_entropy: float | None = None,
    max_damage: float | None = None,
    completed: int | None = None,
) -> dict:
    row = {
        "fidelity": fidelity,
        "milestone_games": milestone,
        "completed_games": completed if completed is not None else milestone,
        "status": "running",
        "metrics": {},
    }
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

    def test_silent_without_baseline_band(self) -> None:
        # a young run still inside 0-30k has no baseline to drift from
        rows = [
            _row(10_000, avg_game_length=45.0),
            _row(20_000, avg_game_length=60.0),
        ]
        assert "game_length_drift" not in _alarm_names(rows)

    def test_silent_while_inside_baseline_band(self) -> None:
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
        (alarm,) = [
            a for a in mp.evaluate_watchdogs(rows) if a["watchdog"] == "game_length_drift"
        ]
        assert alarm["latest_avg_game_length"] == 56.0
        assert alarm["baseline_avg_game_length"] == 28.0
        assert alarm["milestone_games"] == 200_000
        assert alarm["threshold_ratio"] == mp.GAME_LENGTH_DRIFT_RATIO
        assert "message" in alarm


class TestStrengthRegression:
    def test_alarm_at_ten_point_drop(self) -> None:
        rows = _baseline_rows() + [
            _row(200_000, max_damage=0.80),
            _row(300_000, max_damage=0.70),  # exactly -10 points from the peak
        ]
        assert "strength_regression" in _alarm_names(rows)

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
        (alarm,) = [
            a for a in mp.evaluate_watchdogs(rows) if a["watchdog"] == "strength_regression"
        ]
        assert alarm["peak_max_damage_win_rate"] == 0.80
        assert alarm["peak_milestone_games"] == 200_000
        assert alarm["latest_max_damage_win_rate"] == 0.69

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
    def _status(iterations: int, gpi: int = 1600, started_from: int = 0) -> dict:
        return {
            "run_id": "foundation-test",
            "status": "running",
            "games_per_iteration": gpi,
            "started_from_completed_games": started_from,
            "latest_checkpoint_path": f"/x/run/iteration-{iterations:04d}/transformer-policy.pt",
            "completed_iterations": [
                {
                    "iteration": i,
                    "checkpoint_path": f"/x/run/iteration-{i:04d}/transformer-policy.pt",
                }
                for i in range(1, iterations + 1)
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
