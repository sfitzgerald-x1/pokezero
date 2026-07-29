"""Unit tests for the strict transition matcher's component comparator.

The comparator decides every divergence verdict in the acceptance measurement,
so its tolerances are pinned here directly rather than only through end-to-end
census numbers. A lenient comparator produces clean aggregates, which is exactly
the failure mode that looks like success.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from engine_transition_differential import (  # noqa: E402
    damage_components,
    roll_components_agree,
)


class HealToFullTolerance(unittest.TestCase):
    """`*_to_full` heals cap in the OPPOSITE direction to `capped_lethal`.

    A heal that tops the mon out restores ``maxhp - hp_before``, so a LARGER
    preceding damage roll makes the heal LARGER. Sharing the one-sided
    ``obs <= eng + 1`` test with `capped_lethal` inverted this class: it rejected
    the real Rest case and accepted a heal 24x too small.
    """

    def test_motivating_rest_case_agrees(self):
        # seed 1310001 step 72: Showdown healed 251 from 2 HP, the engine healed
        # 247 from 6 HP. Same Rest, different Surf roll on the preceding hit.
        observed = [("", -251), ("heal_to_full", 251)]
        engine = [("", -247), ("heal_to_full", 247)]
        self.assertTrue(roll_components_agree(observed, engine, None))

    def test_absurdly_small_heal_is_rejected(self):
        # The reviewer's counter-case: 10 vs 247 is 24x too small and must not
        # pass. Under the old one-sided test it did.
        observed = [("", -251), ("heal_to_full", 10)]
        engine = [("", -247), ("heal_to_full", 247)]
        self.assertFalse(roll_components_agree(observed, engine, None))

    def test_absurdly_large_heal_is_rejected(self):
        observed = [("", -251), ("heal_to_full", 900)]
        engine = [("", -247), ("heal_to_full", 247)]
        self.assertFalse(roll_components_agree(observed, engine, None))

    def test_window_scales_with_the_preceding_roll(self):
        # With no preceding damage there is no spread to absorb, so a heal must
        # match within flooring slack.
        self.assertTrue(roll_components_agree(
            [("heal_to_full", 100)], [("heal_to_full", 101)], None))
        self.assertFalse(roll_components_agree(
            [("heal_to_full", 100)], [("heal_to_full", 140)], None))


class CappedLethalTolerance(unittest.TestCase):
    """A residual that KILLED was clipped by remaining HP: it can only shrink."""

    def test_clipped_residual_below_engine_agrees(self):
        self.assertTrue(roll_components_agree(
            [("capped_lethal", -20)], [("capped_lethal", -26)], None))

    def test_residual_larger_than_engine_is_rejected(self):
        self.assertFalse(roll_components_agree(
            [("capped_lethal", -40)], [("capped_lethal", -26)], None))


class ComponentExtraction(unittest.TestCase):
    def test_pre_state_seeds_the_first_delta(self):
        """Without the seed the step's PRIMARY move damage is silently dropped."""

        lines = ["|move|p1a: A|Surf|p2a: B", "|-damage|p2a: B|112/245"]
        unseeded = damage_components(lines)
        self.assertEqual(unseeded["p2"], [])
        seeded = damage_components(lines, {"p1": 300, "p2": 245})
        self.assertEqual(seeded["p2"], [("", -133)])

    def test_zero_delta_components_are_dropped(self):
        """The engine emits `Heal 0` where Showdown emits `|-fail|` and no line."""

        lines = ["|-heal|p2a: B|245/245"]
        self.assertEqual(damage_components(lines, {"p2": 245})["p2"], [])

    def test_heal_that_tops_out_is_tagged_to_full(self):
        lines = ["|-heal|p1a: A|253/253 slp"]
        self.assertEqual(
            damage_components(lines, {"p1": 2})["p1"], [("heal_to_full", 251)]
        )

    def test_partial_heal_keeps_its_exact_tag(self):
        lines = ["|-heal|p1a: A|150/253|[from] item: Leftovers"]
        self.assertEqual(
            damage_components(lines, {"p1": 134})["p1"], [("itemleftovers", 16)]
        )


class LengthMismatch(unittest.TestCase):
    def test_differing_component_counts_never_agree(self):
        self.assertFalse(roll_components_agree([("", -50)], [], None))
        self.assertFalse(roll_components_agree([], [("", -50)], None))


if __name__ == "__main__":
    unittest.main()
