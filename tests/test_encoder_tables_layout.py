"""Contract tests for the schema-bound Rust encoder table export."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_encoder_tables.py"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_encoder_tables_test", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib invariant
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EncoderTablesLayoutTest(unittest.TestCase):
    def test_exported_columns_are_bounded_by_the_declared_schema(self) -> None:
        layout = _load_exporter()._layout_payload()

        self.assertEqual(layout["schema_version"], "pokezero.observation.v2.2")
        self.assertEqual(layout["token_count"], 151)
        self.assertEqual(layout["numeric_feature_count"], 155)
        self.assertTrue(layout["numeric_columns"])
        self.assertTrue(layout["categorical_columns"])
        self.assertLess(max(layout["numeric_columns"].values()), 155)
        self.assertLess(max(layout["categorical_columns"].values()), 51)
        self.assertNotIn("NUMERIC_TT_FAIL", layout["numeric_columns"])

    def test_v3_export_uses_the_grouped_physical_layout(self) -> None:
        exporter = _load_exporter()
        layout = exporter._layout_payload("pokezero.observation.v3")

        self.assertEqual(layout["schema_version"], "pokezero.observation.v3")
        self.assertEqual(layout["token_count"], 151)
        self.assertEqual(layout["numeric_feature_count"], 155)
        self.assertEqual(layout["categorical_feature_count"], 51)
        self.assertLess(max(layout["numeric_columns"].values()), 155)
        self.assertLess(max(layout["categorical_columns"].values()), 51)
        self.assertIn("NUMERIC_TT_FAIL", layout["numeric_columns"])
        self.assertIn("NUMERIC_TT_CONFUSION_SELFHIT", layout["numeric_columns"])
        self.assertNotIn("NUMERIC_SELF_SCREENS", layout["numeric_columns"])
        self.assertNotEqual(
            layout["numeric_columns"]["NUMERIC_BASE_POWER"],
            exporter.showdown.NUMERIC_BASE_POWER,
        )
        self.assertEqual(
            layout["constants"]["boost_stat_slots"][0][1],
            layout["numeric_columns"]["NUMERIC_BOOST_ATK"],
        )
        self.assertEqual(
            layout["constants"]["weather_reveal_order"],
            ["raindance", "sunnyday", "sandstorm"],
        )
        self.assertEqual(layout["constants"]["timed_condition_slots"], [])

    def test_exported_offsets_and_default_masks_match_the_runtime_contract(self) -> None:
        layout = _load_exporter()._layout_payload()

        self.assertEqual(
            layout["token_offsets"],
            {
                "field": 0,
                "self_pokemon": 1,
                "opponent_pokemon": 7,
                "action_candidates": 13,
                "stats": 22,
                "transition": 23,
            },
        )
        self.assertEqual(
            layout["default_feature_masks"],
            {
                "stats_block": True,
                "exact_state": True,
                "transition_token_budget": 128,
                "tier2_residuals": True,
                "tier2_investment": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
