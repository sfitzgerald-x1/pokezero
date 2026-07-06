from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pokezero.behavior_metrics import classify_move, move_class_summary
from pokezero.diversity_population import diversity_population_dashboard, payoff_effective_rank
from pokezero.hazard_metrics import aggregate_hazard_rows, correct_pricing, parse_milestone_games_text


ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "hazard_trajectory.py"
_SPEC = importlib.util.spec_from_file_location("hazard_trajectory", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
hazard_trajectory = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hazard_trajectory)

_POP_SCRIPT = ROOT / "scripts" / "diversity_population_dashboard.py"
_POP_SPEC = importlib.util.spec_from_file_location("diversity_population_dashboard", _POP_SCRIPT)
assert _POP_SPEC is not None and _POP_SPEC.loader is not None
diversity_population_dashboard_script = importlib.util.module_from_spec(_POP_SPEC)
_POP_SPEC.loader.exec_module(diversity_population_dashboard_script)


class _FakeDex:
    def move_info(self, move):
        move_id = "".join(ch for ch in str(move).lower() if ch.isalnum())
        if move_id == "thunderbolt":
            return SimpleNamespace(base_power=95, gen3_category="Special", heal=False, status=None, effect_label="")
        if move_id == "protect":
            return SimpleNamespace(base_power=0, gen3_category="Status", heal=False, status=None, effect_label="")
        if move_id == "recover":
            return SimpleNamespace(base_power=0, gen3_category="Status", heal=True, status=None, effect_label="")
        return None


class BehaviorMetricTest(unittest.TestCase):
    def test_classifies_observed_moves_into_measurement_buckets(self) -> None:
        dex = _FakeDex()

        self.assertEqual(classify_move("Spikes", dex=dex), "hazard")
        self.assertEqual(classify_move("Rapid Spin", dex=dex), "clear")
        self.assertEqual(classify_move("Swords Dance", dex=dex), "setup")
        self.assertEqual(classify_move("Thunder Wave", dex=dex), "status")
        self.assertEqual(classify_move("Recover", dex=dex), "heal")
        self.assertEqual(classify_move("Roar", dex=dex), "phaze")
        self.assertEqual(classify_move("Thunderbolt", dex=dex), "attack")
        self.assertEqual(classify_move("Protect", dex=dex), "other")

    def test_move_class_summary_reports_counts_and_rates(self) -> None:
        summary = move_class_summary(
            {"Spikes": 2, "Rapid Spin": 1, "Thunderbolt": 1, "Protect": 1},
            dex=_FakeDex(),
        )

        self.assertEqual(summary["hazard"], {"count": 2, "rate": 0.4})
        self.assertEqual(summary["clear"], {"count": 1, "rate": 0.2})
        self.assertEqual(summary["attack"], {"count": 1, "rate": 0.2})
        self.assertEqual(summary["other"], {"count": 1, "rate": 0.2})


class DiversityPopulationDashboardTest(unittest.TestCase):
    def test_behavior_dashboard_counts_live_strategy_axes(self) -> None:
        rows = [
            {
                "label": "hazard-agent",
                "move_class_usage": {
                    "hazard": {"count": 20, "rate": 0.25},
                    "clear": {"count": 0, "rate": 0.0},
                    "setup": {"count": 1, "rate": 0.0125},
                    "status": {"count": 4, "rate": 0.05},
                    "heal": {"count": 2, "rate": 0.025},
                    "phaze": {"count": 0, "rate": 0.0},
                    "attack": {"count": 53, "rate": 0.6625},
                    "other": {"count": 0, "rate": 0.0},
                },
                "pivot_rate": 0.2,
                "avg_turns": 68.0,
                "distinct_moves": 19,
            },
            {
                "label": "tempo-agent",
                "move_class_usage": {
                    "hazard": {"count": 0, "rate": 0.0},
                    "clear": {"count": 0, "rate": 0.0},
                    "setup": {"count": 8, "rate": 0.08},
                    "status": {"count": 0, "rate": 0.0},
                    "heal": {"count": 0, "rate": 0.0},
                    "phaze": {"count": 4, "rate": 0.04},
                    "attack": {"count": 88, "rate": 0.88},
                    "other": {"count": 0, "rate": 0.0},
                },
                "pivot_rate": 0.02,
                "avg_turns": 42.0,
                "distinct_moves": 11,
            },
        ]

        payload = diversity_population_dashboard(rows)

        self.assertEqual(payload["schema_version"], "pokezero.diversity_population_dashboard.v1")
        self.assertGreaterEqual(payload["behavior"]["live_axis_count"], 4)
        self.assertTrue(payload["behavior"]["axes"]["hazard_cycle"]["live_spread"])
        self.assertTrue(payload["behavior"]["axes"]["tempo"]["live_spread"])
        self.assertTrue(payload["behavior"]["axes"]["aggression_structure"]["live_spread"])
        self.assertTrue(payload["behavior"]["axes"]["interaction"]["live_spread"])
        hazard_metric = payload["behavior"]["axes"]["hazard_cycle"]["metrics"]["move_class_rate:hazard"]
        self.assertEqual(hazard_metric["min_label"], "tempo-agent")
        self.assertEqual(hazard_metric["max_label"], "hazard-agent")
        self.assertEqual(hazard_metric["spread"], 0.25)

    def test_payoff_effective_rank_distinguishes_duplicate_and_independent_vectors(self) -> None:
        duplicate = payoff_effective_rank(
            {
                "a": {"x": 0.8, "y": 0.5},
                "b": {"x": 0.8, "y": 0.5},
            }
        )
        independent = payoff_effective_rank(
            {
                "a": {"x": 0.8, "y": 0.5},
                "b": {"x": 0.5, "y": 0.8},
            }
        )

        self.assertEqual(duplicate["linear_rank"], 1)
        self.assertEqual(duplicate["effective_rank"], 1.0)
        self.assertEqual(independent["linear_rank"], 2)
        self.assertEqual(independent["effective_rank"], 2.0)

    def test_payoff_effective_rank_treats_bad_values_as_neutral(self) -> None:
        payload = payoff_effective_rank(
            {
                "a": {"x": "bad", "y": 0.8},
                "b": {"x": 0.8, "y": 0.5},
            }
        )

        self.assertEqual(payload["member_count"], 2)
        self.assertGreaterEqual(payload["linear_rank"], 1)

    def test_population_dashboard_cli_reads_behavior_and_pool_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            behavior = root / "behavior.json"
            ledger = root / "ledger.json"
            out = root / "dashboard.json"
            behavior.write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {
                                "label": "a",
                                "move_class_usage": {"hazard": {"rate": 0.2}, "attack": {"rate": 0.6}},
                                "pivot_rate": 0.2,
                                "avg_turns": 60,
                                "distinct_moves": 12,
                            },
                            {
                                "label": "b",
                                "move_class_usage": {"hazard": {"rate": 0.0}, "attack": {"rate": 0.9}},
                                "pivot_rate": 0.01,
                                "avg_turns": 40,
                                "distinct_moves": 8,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            ledger.write_text(
                json.dumps(
                    {
                        "payoff_vectors": {
                            "a": {"b": 0.8},
                            "b": {"a": 0.8},
                        }
                    }
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = diversity_population_dashboard_script.main(
                    [
                        "--behavior",
                        str(behavior),
                        "--pool-ledger",
                        str(ledger),
                        "--out",
                        str(out),
                    ]
                )
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("[diversity-population] wrote", stderr.getvalue())
        self.assertTrue(payload["behavior"]["axes"]["hazard_cycle"]["live_spread"])
        self.assertEqual(payload["payoff_rank"]["member_count"], 2)


class HazardTrajectoryTest(unittest.TestCase):
    def test_correct_pricing_uses_design_formula(self) -> None:
        row = {
            "value_spread": 0.5,
            "value_self_hazard_response": -0.05,
            "value_opp_hazard_response": 0.10,
        }

        self.assertEqual(correct_pricing(row), 0.3)

    def test_aggregates_gate_from_milestone_rows(self) -> None:
        rows = []
        for index, pricing in enumerate((0.05, 0.08, 0.11, 0.14, 0.18), start=1):
            rows.append(
                {
                    "label": f"{index * 50}k",
                    "value_spread": 1.0,
                    "value_self_hazard_response": -pricing / 2,
                    "value_opp_hazard_response": pricing / 2,
                    "spin_hazard_response": 0.0 if index < 4 else 0.01,
                }
            )

        payload = aggregate_hazard_rows(rows, threshold=0.10)

        self.assertEqual(payload["valid_points"], 5)
        self.assertTrue(payload["trend_pass"])
        self.assertTrue(payload["last_two_level_pass"])
        self.assertTrue(payload["last_two_correctly_signed"])
        self.assertTrue(payload["last_two_spin_corrob"])
        self.assertTrue(payload["spin_corrob"])
        self.assertTrue(payload["gate_pass"])
        self.assertEqual([point["milestone_games"] for point in payload["points"]], [50_000, 100_000, 150_000, 200_000, 250_000])

    def test_gate_requires_spin_corroboration_in_flip_window(self) -> None:
        rows = []
        for index, pricing in enumerate((0.05, 0.08, 0.11, 0.14, 0.18), start=1):
            rows.append(
                {
                    "milestone_games": index * 50_000,
                    "value_spread": 1.0,
                    "value_self_hazard_response": -pricing / 2,
                    "value_opp_hazard_response": pricing / 2,
                    "spin_hazard_response": 0.01 if index == 1 else 0.0,
                }
            )

        payload = aggregate_hazard_rows(rows, threshold=0.10)

        self.assertFalse(payload["last_two_spin_corrob"])
        self.assertFalse(payload["gate_pass"])

    def test_gate_fails_without_valid_spread(self) -> None:
        payload = aggregate_hazard_rows(
            [
                {
                    "label": "50k",
                    "value_spread": 0.0,
                    "value_self_hazard_response": -0.1,
                    "value_opp_hazard_response": 0.1,
                    "spin_hazard_response": 0.1,
                }
            ]
        )

        self.assertEqual(payload["valid_points"], 0)
        self.assertFalse(payload["gate_pass"])

    def test_zero_game_milestone_sorts_before_later_points(self) -> None:
        payload = aggregate_hazard_rows(
            [
                {
                    "milestone_games": 0,
                    "value_spread": 1.0,
                    "value_self_hazard_response": -0.01,
                    "value_opp_hazard_response": 0.01,
                },
                {
                    "milestone_games": 30_000,
                    "value_spread": 1.0,
                    "value_self_hazard_response": -0.04,
                    "value_opp_hazard_response": 0.04,
                },
                {
                    # This second zero milestone would be misordered by
                    # `milestone_games or index` because its row index is
                    # greater than the later one-game milestone.
                    "milestone_games": 0,
                    "value_spread": 1.0,
                    "value_self_hazard_response": -0.02,
                    "value_opp_hazard_response": 0.02,
                },
                {
                    "milestone_games": 1,
                    "value_spread": 1.0,
                    "value_self_hazard_response": -0.03,
                    "value_opp_hazard_response": 0.03,
                },
            ]
        )

        self.assertEqual([point["milestone_games"] for point in payload["points"]], [0, 0, 1, 30_000])

    def test_milestone_parser_handles_decimal_millions_and_rejects_dates(self) -> None:
        self.assertEqual(parse_milestone_games_text("pokezero-belief-gen3-1-5m"), 1_500_000)
        self.assertEqual(parse_milestone_games_text("1.25M"), 1_250_000)
        self.assertEqual(parse_milestone_games_text("250k"), 250_000)
        self.assertIsNone(parse_milestone_games_text("20260705"))

    def test_gate_fails_when_valid_points_lack_ordering(self) -> None:
        rows = []
        for index, pricing in enumerate((0.05, 0.08, 0.11, 0.14, 0.18), start=1):
            rows.append(
                {
                    "label": f"checkpoint-{index}",
                    "value_spread": 1.0,
                    "value_self_hazard_response": -pricing / 2,
                    "value_opp_hazard_response": pricing / 2,
                    "spin_hazard_response": 0.01,
                }
            )

        payload = aggregate_hazard_rows(rows, threshold=0.10)

        self.assertFalse(payload["ordering_complete"])
        self.assertEqual(payload["missing_milestone_point_indexes"], [0, 1, 2, 3, 4])
        self.assertFalse(payload["gate_pass"])

    def test_hazard_trajectory_cli_infers_milestones_from_sweep_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hazards = []
            for index, pricing in enumerate((0.05, 0.08, 0.11, 0.14, 0.18), start=1):
                path = root / f"hazard-{index * 50_000}.json"
                path.write_text(
                    json.dumps(
                        {
                            "checkpoints": [
                                {
                                    # milestone_probes.sh labels rows as
                                    # <run>-i<iteration>, so filename is the
                                    # only durable milestone source.
                                    "label": f"cycle-a-main-i{index}",
                                    "value_spread": 1.0,
                                    "value_self_hazard_response": -pricing / 2,
                                    "value_opp_hazard_response": pricing / 2,
                                    "spin_hazard_response": 0.0 if index < 4 else 0.01,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                hazards.append(path)
            out = root / "trajectory.json"

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = hazard_trajectory.main(
                    [
                        *(arg for hazard in hazards for arg in ("--hazard", str(hazard))),
                        "--out",
                        str(out),
                    ]
                )
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("[hazard-trajectory] wrote", stderr.getvalue())
        self.assertTrue(payload["ordering_complete"])
        self.assertTrue(payload["gate_pass"])
        self.assertEqual(
            [point["milestone_games"] for point in payload["points"]],
            [50_000, 100_000, 150_000, 200_000, 250_000],
        )


if __name__ == "__main__":
    unittest.main()
