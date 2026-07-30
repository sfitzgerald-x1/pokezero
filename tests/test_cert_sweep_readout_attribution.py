"""Focused pins for certification-sweep attribution boundaries."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from cert_sweep_readout import attribute_row  # noqa: E402


def _row(divergence_class: str, *misses: str) -> dict:
    return {
        "divergence_class": divergence_class,
        "branch_misses": list(misses),
        "protocol": ["|upkeep"],
        "choices": {},
    }


class CertificationAttributionTests(unittest.TestCase):
    def test_i4_accepts_majority_arm_label_tie(self) -> None:
        row = _row(
            "component_mismatch:heal|itemleftovers",
            "pct=75.00: p1 attributed components differ: observed_only=[('heal', 10)] "
            "engine_only=[('itemleftovers', 10)]",
            "pct=25.00: p1 attributed components differ: observed_only=[('heal', 10)] "
            "engine_only=[('heal', 8)]",
        )
        self.assertEqual(attribute_row(row)[0], "I4_attribution_tie")

    def test_i4_rejects_tie_confined_to_minority_arm(self) -> None:
        row = _row(
            "component_magnitude:heal",
            "pct=75.00: p1 attributed components differ: observed_only=[('heal', 10)] "
            "engine_only=[('heal', 8)]",
            "pct=25.00: p1 attributed components differ: observed_only=[('heal', 10)] "
            "engine_only=[('itemleftovers', 10)]",
        )
        self.assertNotEqual(attribute_row(row)[0], "I4_attribution_tie")

    def test_structural_echo_requires_exact_sibling_components(self) -> None:
        row = _row(
            "roll_scaled_component",
            "pct=75.00: p1 roll-scaled components differ: "
            "observed_only=[('', -10), ('recoil', -3)] "
            "engine_only=[('', -10)]",
            "pct=25.00: p1 roll-scaled components differ: "
            "observed_only=[('', -10), ('recoil', -3)] "
            "engine_only=[('', -10), ('recoil', -3)]",
        )
        self.assertEqual(attribute_row(row)[0], "LS_structural_arm_echo")

    def test_structural_echo_rejects_unsupported_count_mismatch(self) -> None:
        row = _row(
            "roll_scaled_component",
            "pct=75.00: p1 roll-scaled components differ: "
            "observed_only=[('', -10), ('recoil', -3)] "
            "engine_only=[('', -10)]",
            "pct=25.00: p1 roll-scaled components differ: "
            "observed_only=[('', -10), ('recoil', -3)] "
            "engine_only=[('', -9), ('recoil', -3)]",
        )
        family, basis = attribute_row(row)
        self.assertEqual(family, "UNATTRIBUTED")
        self.assertIn("without a sibling engine arm", basis)

    def test_structural_echo_rejects_opposite_side_sibling(self) -> None:
        row = _row(
            "roll_scaled_component",
            "pct=75.00: p1 roll-scaled components differ: "
            "observed_only=[('', -10), ('recoil', -3)] "
            "engine_only=[('', -10)]",
            "pct=25.00: p2 roll-scaled components differ: "
            "observed_only=[('', -10), ('recoil', -3)] "
            "engine_only=[('', -10), ('recoil', -3)]",
        )
        family, basis = attribute_row(row)
        self.assertEqual(family, "UNATTRIBUTED")
        self.assertIn("without a sibling engine arm", basis)


if __name__ == "__main__":
    unittest.main()
