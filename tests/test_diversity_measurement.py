from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from pokezero.behavior_metrics import classify_move, move_class_summary


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

        self.assertEqual(hazard_trajectory.correct_pricing(row), 0.3)

    def test_aggregates_gate_from_milestone_rows(self) -> None:
        rows = []
        for index, pricing in enumerate((0.05, 0.08, 0.11, 0.14, 0.18), start=1):
            rows.append(
                {
                    "label": f"{index * 50}k",
                    "value_spread": 1.0,
                    "value_self_hazard_response": -pricing / 2,
                    "value_opp_hazard_response": pricing / 2,
                    "spin_hazard_response": 0.0 if index < 5 else 0.01,
                }
            )

        payload = hazard_trajectory.aggregate_hazard_rows(rows, threshold=0.10)

        self.assertEqual(payload["valid_points"], 5)
        self.assertTrue(payload["monotone_non_decreasing"])
        self.assertTrue(payload["last_two_level_pass"])
        self.assertTrue(payload["last_two_correctly_signed"])
        self.assertTrue(payload["spin_corrob"])
        self.assertTrue(payload["gate_pass"])
        self.assertEqual([point["milestone_games"] for point in payload["points"]], [50_000, 100_000, 150_000, 200_000, 250_000])

    def test_gate_fails_without_valid_spread(self) -> None:
        payload = hazard_trajectory.aggregate_hazard_rows(
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


if __name__ == "__main__":
    unittest.main()
