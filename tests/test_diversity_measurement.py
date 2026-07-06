from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from pokezero.behavior_metrics import classify_move, move_class_summary
from pokezero.hazard_metrics import aggregate_hazard_rows, correct_pricing, parse_milestone_games_text


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hazard_trajectory.py"
_SPEC = importlib.util.spec_from_file_location("hazard_trajectory", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
hazard_trajectory = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hazard_trajectory)


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
                    "milestone_games": 50_000,
                    "value_spread": 1.0,
                    "value_self_hazard_response": -0.04,
                    "value_opp_hazard_response": 0.04,
                },
                {
                    "milestone_games": 0,
                    "value_spread": 1.0,
                    "value_self_hazard_response": -0.02,
                    "value_opp_hazard_response": 0.02,
                },
            ]
        )

        self.assertEqual([point["milestone_games"] for point in payload["points"]], [0, 50_000])

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


if __name__ == "__main__":
    unittest.main()
