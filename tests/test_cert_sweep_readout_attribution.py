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


class NarrowingPinTests(unittest.TestCase):
    """Mutation pins for the three narrowings in this change.

    Re-review mutation-tested all three and found every one reverts silently:
    flipping either `for miss in [majority]` back to `for miss in misses`, or
    the Truant predicate back to `"Slaking" in proto`, left the whole cert suite
    green. The PR's only behavioural content was undefended. Each test below is
    the minority-arm-decides case that the un-narrowed rule absorbs and the
    narrowed rule must not.
    """

    @staticmethod
    def _row(divergence_class, misses, protocol=("|upkeep",)):
        return {
            "divergence_class": divergence_class,
            "branch_misses": list(misses),
            "protocol": list(protocol),
            "choices": {},
        }

    # -- rule 1: same-turn boost/status magnitude ratio ---------------------
    #    A 6% arm carries the boost-ratio shape; the 94% MAJORITY arm carries a
    #    plain count mismatch. Scanning every arm lets the minority decide.
    _BOOST_MINORITY = (
        "pct=94.00: p1 roll-scaled components differ: "
        "observed_only=[] engine_only=[('', -40)]",
        "pct=6.00: p1 roll-scaled components differ: "
        "observed_only=[('', -80)] engine_only=[('', -100)]",
    )

    def test_a_minority_boost_ratio_arm_does_not_decide_the_row(self) -> None:
        row = self._row(
            "roll_scaled_component", self._BOOST_MINORITY, ("|-boost|p1a: X|atk|1",)
        )
        family, basis = attribute_row(row)
        self.assertNotIn("magnitude ratio", basis, basis)

    def test_the_boost_ratio_rule_still_fires_on_the_majority_arm(self) -> None:
        row = self._row(
            "roll_scaled_component",
            (
                "pct=94.00: p1 roll-scaled components differ: "
                "observed_only=[('', -80)] engine_only=[('', -100)]",
            ),
            ("|-boost|p1a: X|atk|1",),
        )
        _, basis = attribute_row(row)
        self.assertIn("magnitude ratio", basis, basis)

    # -- rule 2: roll_scaled_component structural count ---------------------
    def test_a_minority_structural_arm_does_not_decide_the_row(self) -> None:
        """The majority arm has EQUAL counts, so no structural claim applies."""

        row = self._row(
            "roll_scaled_component",
            (
                "pct=94.00: p1 roll-scaled components differ: "
                "observed_only=[('', -80)] engine_only=[('', -100)]",
                "pct=6.00: p1 roll-scaled components differ: "
                "observed_only=[] engine_only=[('sandstorm', -6)]",
            ),
        )
        _, basis = attribute_row(row)
        self.assertNotIn("structural component-count mismatch", basis, basis)

    # -- rule 3: Truant loaf phase, BOTH directions -------------------------
    _TRUANT_MISS = (
        "pct=100.00: p1 roll-scaled components differ: "
        "observed_only=[] engine_only=[('', -10)]",
    )
    _TRUANT_MISS_OBS = (
        "pct=100.00: p1 roll-scaled components differ: "
        "observed_only=[('', -10)] engine_only=[]",
    )

    def test_a_slaking_elsewhere_in_the_protocol_does_not_qualify(self) -> None:
        """The 8-of-9 defect: a switch boundary whose complaining slot holds a
        different mon, in a game that merely contains a Slaking."""

        row = self._row(
            "roll_scaled_component",
            self._TRUANT_MISS,
            ("|switch|p2a: Slaking|Slaking|100/100", "|move|p1a: Blissey|seismictoss"),
        )
        _, basis = attribute_row(row)
        self.assertNotIn("loaf-phase", basis, basis)

    def test_the_loafing_direction_qualifies(self) -> None:
        row = self._row(
            "roll_scaled_component",
            self._TRUANT_MISS,
            ("|cant|p1a: Slaking|ability: Truant",),
        )
        _, basis = attribute_row(row)
        self.assertIn("loaf-phase", basis, basis)

    def test_the_attacking_direction_also_qualifies(self) -> None:
        """s2000059/11 -- the rule's PRIMARY cited validation row. The Slaking
        attacks in the sim and only the engine's branch loafed, so there is no
        |cant| line anywhere. The first narrowing dropped this direction."""

        row = self._row(
            "roll_scaled_component",
            self._TRUANT_MISS_OBS,
            ("|move|p1a: Slaking|return|p2a: Blissey",),
        )
        _, basis = attribute_row(row)
        self.assertIn("loaf-phase", basis, basis)
