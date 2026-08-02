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
        # Either side: what matters is that a Truant holder ACTED, in subject
        # position. An earlier version required the Slaking opposite the
        # complaining slot, which dropped the same-side recoil shape.
        cases["truant_loaf_phase_drift"]["protocol"] = [
            "|cant|p2a: Slaking|ability: Truant"
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

    # -- rule 3: Truant loaf phase ------------------------------------------
    #    The rule keys on a Truant holder ACTING in subject position, on
    #    EITHER side. Two earlier versions keyed on which side the Slaking was
    #    on -- first the complaining slot, then the opposite one -- and each
    #    dropped a real shape while its tests stayed green, because the pins
    #    used the same wrong frame as the code.
    _TRUANT_MISS = (
        "pct=100.00: p1 roll-scaled components differ: "
        "observed_only=[] engine_only=[('', -10)]",
    )
    _TRUANT_MISS_OBS = (
        "pct=100.00: p1 roll-scaled components differ: "
        "observed_only=[('', -130)] engine_only=[]",
    )

    def test_a_slaking_elsewhere_in_the_protocol_does_not_qualify(self) -> None:
        """The 8-of-9 defect: a game that merely contains a Slaking."""

        row = self._row(
            "roll_scaled_component",
            self._TRUANT_MISS,
            ("|switch|p2a: Slaking|Slaking|100/100", "|move|p1a: Blissey|seismictoss"),
        )
        _, basis = attribute_row(row)
        self.assertNotIn("loaf-phase", basis, basis)

    def test_the_loafing_direction_qualifies(self) -> None:
        """Sim's Slaking loafed; the engine's branch attacked p1."""

        row = self._row(
            "roll_scaled_component",
            self._TRUANT_MISS,
            ("|cant|p2a: Slaking|ability: Truant",),
        )
        _, basis = attribute_row(row)
        self.assertIn("loaf-phase", basis, basis)

    def test_the_attacking_direction_also_qualifies(self) -> None:
        """s2000059/11 -- the rule's PRIMARY cited validation row, and the real
        retained shape: p1a Slaking hits p2a for -130 and the miss complains
        about p2. Mirrored here as p2a Slaking hitting p1."""

        row = self._row(
            "roll_scaled_component",
            self._TRUANT_MISS_OBS,
            ("|move|p2a: Slaking|shadowball|p1a: Zapdos",),
        )
        _, basis = attribute_row(row)
        self.assertIn("loaf-phase", basis, basis)

    def test_a_slaking_that_acted_then_fainted_still_qualifies(self) -> None:
        """Round eight asserted the opposite; round nine superseded it.

        Round eight read this as "the switch-boundary shape" because the
        complaining slot ends up holding a different mon. But the Slaking
        ACTED -- it used Double-Edge -- and a one-sided component difference on
        a turn where a Truant holder took its turn is exactly the loaf-phase
        question. What the rule must reject is a Slaking that never acted, and
        subject-position matching is what draws that line.

        The original 8-of-9 defect stays rejected for the right reason and is
        pinned separately: `|switch|p2a: Slaking` is not a subject-position
        `|move|`/`|cant|`, and neither is a Slaking named only as a target.
        """

        row = self._row(
            "roll_scaled_component",
            self._TRUANT_MISS_OBS,
            (
                "|move|p1a: Slaking|doubleedge|p2a: Zapdos",
                "|-damage|p1a: Slaking|0 fnt",
                "|faint|p1a: Slaking",
                "|switch|p1a: Blissey|Blissey|100/100",
            ),
        )
        _, basis = attribute_row(row)
        self.assertIn("loaf-phase", basis, basis)

    _RECOIL_MISS = (
        "pct=100.00: p1 roll-scaled components differ: "
        "observed_only=[('recoil', -30)] engine_only=[]",
    )

    def test_a_same_side_recoil_component_still_qualifies(self) -> None:
        """Round nine: the side restriction created a FALSE NEGATIVE.

        The differential breaks on the first disagreeing slot in the fixed
        order (p1, p2), so a p1 Slaking whose Double-Edge leaves same-side
        recoil makes p1 the complaining slot. An attacker-side rule computed p2,
        found nothing, and dropped a genuine loaf drift that main attributed.
        Double-Edge is a standard gen3 Slaking move, so this bites in practice.
        """

        row = self._row(
            "roll_scaled_component",
            self._RECOIL_MISS,
            ("|move|p1a: Slaking|doubleedge|p2a: Zapdos",),
        )
        _, basis = attribute_row(row)
        self.assertIn("loaf-phase", basis, basis)

    def test_a_slaking_immobilised_for_some_other_reason_does_not_qualify(self) -> None:
        """A |cant| must be a TRUANT loaf, not any immobilisation.

        Paralysis, sleep and flinch all emit |cant| for the same Slaking. Only
        Truant is the loaf phase this counter predicts zero for; treating every
        |cant| as a loaf would re-widen the rule through a different door.
        """

        for reason in ("par", "slp", "flinch"):
            with self.subTest(reason=reason):
                row = self._row(
                    "roll_scaled_component",
                    self._TRUANT_MISS,
                    (f"|cant|p2a: Slaking|{reason}",),
                )
                _, basis = attribute_row(row)
                self.assertNotIn("loaf-phase", basis, basis)

    def test_a_slaking_that_is_only_a_move_target_does_not_qualify(self) -> None:
        """Round nine: the residual FALSE POSITIVE of the exact class this rule
        was narrowed to reject.

        `_actor in line` matched the Slaking as the move's TARGET, so a Slaking
        being phazed off the field fired loaf-phase without ever acting -- and
        firing manufactures a certification failure through the registered-zero
        gate. Subject position is what says "the Slaking acted".
        """

        row = self._row(
            "roll_scaled_component",
            self._RECOIL_MISS,
            (
                "|move|p1a: Skarmory|whirlwind|p2a: Slaking",
                "|drag|p2a: Blissey|Blissey|100/100",
            ),
        )
        _, basis = attribute_row(row)
        self.assertNotIn("loaf-phase", basis, basis)


class TracedTruantTests(unittest.TestCase):
    """A Porygon2 that Traces Slaking's Truant is still a Truant holder.

    Round ten measured this as a coverage regression against main: the species
    gate on the |cant| arm dropped it. The ability tag already proves the
    mechanism, so the species is redundant there. docs ledger 6212 states the
    rule with no species term.
    """

    @staticmethod
    def _row(protocol):
        return {
            "divergence_class": "roll_scaled_component",
            "branch_misses": [
                "pct=100.00: p1 roll-scaled components differ: "
                "observed_only=[] engine_only=[('', -10)]"
            ],
            "protocol": list(protocol),
            "choices": {},
        }

    def test_a_traced_truant_loaf_qualifies(self) -> None:
        _, basis = attribute_row(self._row(["|cant|p2a: Porygon2|ability: Truant"]))
        self.assertIn("loaf-phase", basis, basis)

    def test_a_native_slaking_loaf_still_qualifies(self) -> None:
        _, basis = attribute_row(self._row(["|cant|p2a: Slaking|ability: Truant"]))
        self.assertIn("loaf-phase", basis, basis)

    def test_a_non_truant_cant_still_does_not(self) -> None:
        _, basis = attribute_row(self._row(["|cant|p2a: Porygon2|par"]))
        self.assertNotIn("loaf-phase", basis, basis)
