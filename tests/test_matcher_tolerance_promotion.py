"""Pins for the three matcher tolerance changes.

These shipped once with no tests at all, and review found two of them admitted
physically impossible pairs. The design here is the reviewer's, which they
implemented and measured at 99.99% recovery of realizable cap-bearing pairs
against 99.89% for the version that shipped -- strictly better AND sound.

The load-bearing measurement: without the promotion, the same 200-game window
measures 208 divergent; with it, 67. The whole 141-row delta is
roll_scaled_component. So these cannot simply be deleted.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from engine_transition_differential import (  # noqa: E402
    DamageComponent,
    _roll_damage_scale,
    _split_component_events,
    roll_components_agree,
)


def comp(source, delta, index=0):
    return DamageComponent(source=source, delta=delta, event_index=index)


class PromotionIsForLengthOnly(unittest.TestCase):
    def test_a_cap_on_one_side_no_longer_changes_the_list_length(self) -> None:
        """The defect promotion exists to fix: cap membership is roll-dependent,
        so two arms differing only by a legal roll read as structurally
        different and are rejected on LENGTH before any tolerance applies."""

        observed = [comp("", -155, 0), comp("itemleftovers_to_full", 20, 1)]
        engine = [comp("", -165, 0), comp("itemleftovers", 20, 1)]
        _, obs_rolled = _split_component_events(observed)
        _, eng_rolled = _split_component_events(engine)
        self.assertNotEqual(len(obs_rolled), len(eng_rolled))

        bases = {"itemleftovers"}
        _, obs_promoted = _split_component_events(observed, capped_bases=bases)
        _, eng_promoted = _split_component_events(engine, capped_bases=bases)
        self.assertEqual(len(obs_promoted), len(eng_promoted))

    def test_a_promoted_but_uncapped_heal_is_compared_exactly(self) -> None:
        """gen3 heal amounts are deterministic -- a bare -heal is
        floor(maxhp/2). A 2 HP discrepancy is the rounding class this program
        exists to find, and the first version let promotion grant it the 9%
        window so 18 and 20 matched."""

        self.assertFalse(
            roll_components_agree(
                [("heal_to_full", 30), ("heal", 20), ("", -100)],
                [("heal_to_full", 30), ("heal", 18), ("", -100)],
                None,
            )
        )

    def test_an_identical_pair_still_agrees(self) -> None:
        self.assertTrue(
            roll_components_agree(
                [("heal_to_full", 30), ("heal", 20), ("", -100)],
                [("heal_to_full", 30), ("heal", 20), ("", -100)],
                None,
            )
        )


class CapToleranceIsTheDamageDifference(unittest.TestCase):
    def test_identical_damage_forbids_any_cap_difference(self) -> None:
        """Both arms took -260, so their deficits are equal and their caps must
        be too. The fraction bound gave 0.18*260+1 = 47.8 HP of slack and
        accepted 40 against 85."""

        self.assertFalse(
            roll_components_agree(
                [("", -260), ("movewish_to_full", 40)],
                [("", -260), ("movewish_to_full", 85)],
                None,
            )
        )

    def test_a_cap_difference_within_the_damage_difference_agrees(self) -> None:
        self.assertTrue(
            roll_components_agree(
                [("", -155), ("heal_to_full", 160)],
                [("", -165), ("heal_to_full", 170)],
                None,
            )
        )

    def test_a_cap_difference_beyond_the_damage_difference_is_rejected(self) -> None:
        self.assertFalse(
            roll_components_agree(
                [("", -155), ("heal_to_full", 160)],
                [("", -165), ("heal_to_full", 200)],
                None,
            )
        )

    def test_the_arm_that_took_more_damage_must_carry_the_larger_cap(self) -> None:
        """A symmetric absolute bound admits the inversion: |160-155| <=
        |165-155| passes, but the arm that took LESS damage cannot have the
        DEEPER deficit."""

        self.assertFalse(
            roll_components_agree(
                [("", -155), ("heal_to_full", 160)],
                [("", -165), ("heal_to_full", 155)],
                None,
            )
        )


if __name__ == "__main__":
    unittest.main()


class DamageDifferenceIsSlotWide(unittest.TestCase):
    """The bound is a property of the SLOT's damage, not of one component.

    This was the bug that cost four wrong attributions. Computing it inside
    roll_components_agree from `observed`/`engine` looks right, but the
    equal-length caller passes ONE component at a time and _roll_damage_scale
    excludes heals -- so for a heal pair the local value is 0 and the bound
    collapses to 1 HP. Far TIGHTER than intended, which is the opposite of how
    I described it for three rounds. Measured effect on a 200-game window:
    208 divergent with the collapsed bound, 64 with the slot-wide one.
    """

    OBS = [("", -155), ("heal_to_full", 160)]
    ENG = [("", -165), ("heal_to_full", 170)]

    def test_the_unthreaded_bound_collapses_and_false_rejects(self) -> None:
        """Without damage_scales, a single heal pair sees 0 damage."""

        self.assertFalse(
            roll_components_agree([self.OBS[1]], [self.ENG[1]], None)
        )

    def test_the_threaded_bound_accepts_the_same_realizable_pair(self) -> None:
        scales = (_roll_damage_scale(self.OBS), _roll_damage_scale(self.ENG))
        self.assertTrue(
            roll_components_agree(
                [self.OBS[1]], [self.ENG[1]], None, damage_scales=scales
            )
        )

    def test_threading_does_not_cost_soundness(self) -> None:
        """Identical damage still forbids any cap difference."""

        obs = [("", -260), ("movewish_to_full", 40)]
        eng = [("", -260), ("movewish_to_full", 85)]
        scales = (_roll_damage_scale(obs), _roll_damage_scale(eng))
        self.assertFalse(
            roll_components_agree(
                [obs[1]], [eng[1]], None, damage_scales=scales
            )
        )
