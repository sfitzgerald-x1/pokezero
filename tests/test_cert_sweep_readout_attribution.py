"""Focused pins for certification-sweep attribution boundaries."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from cert_sweep_readout import EMITTABLE_EXCLUSION_COUNTERS, attribute_row, classify_row  # noqa: E402


def _row(divergence_class: str, *misses: str) -> dict:
    return {
        "divergence_class": divergence_class,
        "branch_misses": list(misses),
        "protocol": ["|upkeep"],
        "choices": {},
    }


class CertificationAttributionTests(unittest.TestCase):
    def test_every_emittable_exclusion_counter_is_reachable_at_minority_mass(self) -> None:
        cases = {
            "recharge_turn_residual_gap": _row(
                "roll_scaled_component",
                "pct=10.00: p1 roll-scaled components differ: observed_only=[] engine_only=[('sandstorm', -1)]",
            ),
            "truant_loaf_phase_drift": _row(
                "roll_scaled_component",
                "pct=10.00: p1 roll-scaled components differ: observed_only=[] engine_only=[('', -10)]",
            ),
            "absorb_through_protect_or_miss": _row(
                "component_mismatch:heal",
                "pct=10.00: p1 attributed components differ: observed_only=[] engine_only=[('abilitywaterabsorb', 20)]",
            ),
            "recoil_vs_substitute_basis": _row(
                "component_magnitude:recoil",
                "pct=10.00: p1 attributed components differ: observed_only=[('recoil', -10)] engine_only=[('recoil', -21)]",
            ),
            "incapacitated_arm_pricing": _row(
                "roll_scaled_component",
                "pct=10.00: p1 roll-scaled components differ: observed_only=[] engine_only=[('', -10)]",
            ),
            "same_turn_stat_event_gap": _row(
                "component_magnitude:damage",
                "pct=10.00: p1 attributed components differ: observed_only=[('', -87)] engine_only=[('', -100)]",
            ),
            "structural_component_count_without_supported_sibling": _row(
                "roll_scaled_component",
                "pct=10.00: p1 roll-scaled components differ: observed_only=[('', -10), ('recoil', -3)] engine_only=[('', -10)]",
            ),
            "unattributed_generic": _row(
                "new_unclassified_shape",
                "pct=10.00: p1 attributed components differ: observed_only=[('x', 1)] engine_only=[('y', 2)]",
            ),
        }
        cases["recharge_turn_residual_gap"]["protocol"] = ["|cant|p1a: Mon|recharge"]
        # The real loaf signature. The old fixture was |-ability|p1a: Slaking|Truant,
        # which carries no |cant| and so is not a loaf at all -- it went stale when
        # C45 narrowed the rule from "protocol mentions a Slaking" to the actual
        # signature, after finding 8 of its 9 sweep rows were switch boundaries.
        cases["truant_loaf_phase_drift"]["protocol"] = [
            "|cant|p1a: Slaking|ability: Truant"
        ]
        cases["absorb_through_protect_or_miss"]["protocol"] = ["|-activate|p1a: Mon|move: Protect"]
        cases["recoil_vs_substitute_basis"]["protocol"] = ["|-end|p2a: Mon|Substitute"]
        cases["incapacitated_arm_pricing"]["protocol"] = ["|cant|p1a: Mon|frz"]
        cases["same_turn_stat_event_gap"]["protocol"] = ["|-boost|p1a: Mon|spa|1"]

        observed = {counter: classify_row(row)[2] for counter, row in cases.items()}
        self.assertEqual(set(observed), set(EMITTABLE_EXCLUSION_COUNTERS))
        self.assertEqual(observed, {counter: counter for counter in EMITTABLE_EXCLUSION_COUNTERS})

    def test_minority_absorb_same_turn_and_structural_shapes_outrank_i2(self) -> None:
        absorb = _row(
            "component_mismatch:heal",
            "pct=10.00: p1 attributed components differ: observed_only=[] engine_only=[('abilitywaterabsorb', 20)]",
        )
        absorb["protocol"] = ["|-activate|p1a: Mon|move: Protect"]
        stat = _row(
            "component_magnitude:damage",
            "pct=10.00: p1 attributed components differ: observed_only=[('', -87)] engine_only=[('', -100)]",
        )
        stat["protocol"] = ["|-boost|p1a: Mon|spa|1"]
        structural = _row(
            "roll_scaled_component",
            "pct=10.00: p1 roll-scaled components differ: observed_only=[('', -10), ('recoil', -3)] engine_only=[('', -10)]",
        )
        for row, counter in (
            (absorb, "absorb_through_protect_or_miss"),
            (stat, "same_turn_stat_event_gap"),
            (structural, "structural_component_count_without_supported_sibling"),
        ):
            family, _, observed_counter = classify_row(row)
            self.assertEqual(family, "UNATTRIBUTED")
            self.assertEqual(observed_counter, counter)

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
