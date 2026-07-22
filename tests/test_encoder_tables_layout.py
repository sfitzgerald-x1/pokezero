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
