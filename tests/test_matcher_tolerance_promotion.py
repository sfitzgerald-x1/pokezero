"""Pins for the three matcher tolerance changes.

These shipped once with no tests at all, and review found two of them admitted
physically impossible pairs. The design here is the reviewer's, which they
implemented and measured at 99.99% recovery of realizable cap-bearing pairs
against 99.89% for the version that shipped -- strictly better AND sound.

Measured decomposition of the 144-row reduction (208 on main -> 64 here), each
by single-variable sweep on one tree:

    damage_scales threading   123 rows
    capped_bases promotion     17 rows
    bound shape                 3 rows
    severity change             0 rows

The threading is the substance. The bound redesign, which took four rounds of
wrong attribution to get right, is worth 3 of 144.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from engine_transition_differential import (  # noqa: E402
    DamageComponent,
    branch_render_is_usable,
    roll_component_events_agree,
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


class ThreadingIsPinnedAtTheCallSite(unittest.TestCase):
    """The 123-row fix, pinned where it can actually be reverted.

    Review: dropping `damage_scales=_slot_damage` at the
    roll_component_events_agree call site passed all 91 tests. The existing
    pins exercise roll_components_agree's SIGNATURE, never the call site, so
    the entire headline fix could be reverted silently.
    """

    def test_the_event_level_comparator_threads_the_slot_damage(self) -> None:
        """A capped heal whose difference exceeds 1 HP but sits inside the
        slot's damage difference. Passes only if the caller threads."""

        observed = [comp("", -155, 0), comp("heal_to_full", 160, 1)]
        engine = [comp("", -165, 0), comp("heal_to_full", 170, 1)]
        self.assertTrue(
            roll_component_events_agree(
                observed, engine, support=None, target_side="side_one",
                pre_legal=None,
            )
        )

    def test_the_cascade_path_also_threads(self) -> None:
        """Same, through the length-mismatch path."""

        observed = [comp("", -155, 0), comp("heal_to_full", 160, 1)]
        engine = [
            comp("", -165, 0),
            comp("heal", 160, 1),
            comp("itemleftovers_to_full", 10, 2),
        ]
        self.assertTrue(
            roll_component_events_agree(
                observed, engine, support=None, target_side="side_one",
                pre_legal=None,
            )
        )


class PromotionDoesNotLeakThroughTheMirror(unittest.TestCase):
    """The cap tolerance must key on EITHER side being capped.

    Review F1: keying on obs_source alone let an observed PLAIN deterministic
    heal match an engine `_to_full` through the +/-9% roll window -- the same
    masking class promotion was supposedly fixed to prevent, alive in the
    mirror direction, and reachable only because promotion puts the plain
    component in the rolled list at all.
    """

    def test_a_plain_observed_heal_cannot_match_a_capped_engine_heal(self) -> None:
        """Identical -100 on both arms, so the deficits are identical: a capped
        engine 30 forces deficit 30, and an observed 33 is impossible."""

        # 31 is inside the deliberate 1 HP flooring slack, so the first
        # genuinely impossible value is 32.
        for observed_value in (32, 33, 40):
            with self.subTest(observed=observed_value):
                self.assertFalse(
                    roll_components_agree(
                        [("", -100), ("itemleftovers", observed_value)],
                        [("", -100), ("itemleftovers_to_full", 30)],
                        None,
                        damage_scales=(100, 100),
                    )
                )

    def test_the_mirror_cascade_still_agrees_when_damage_differs(self) -> None:
        self.assertTrue(
            roll_components_agree(
                [("", -165), ("heal", 170)],
                [("", -155), ("heal_to_full", 160)],
                None,
                damage_scales=(165, 155),
            )
        )


class SeverityIsAnAllowlist(unittest.TestCase):
    """A branch is usable only if EVERY marker it carries is telemetry-only.

    This change shipped with zero coverage and as the exclusion form its own
    comment condemned; reverting it broke no test. An exclusion fails open as
    the renderer grows, which is the whole reason for the allowlist.
    """

    def test_a_clean_render_is_usable(self) -> None:
        self.assertTrue(branch_render_is_usable([]))

    def test_the_two_telemetry_only_markers_are_usable(self) -> None:
        self.assertTrue(branch_render_is_usable(["sleeptalk_called_unidentified"]))
        self.assertTrue(
            branch_render_is_usable(["attract_immobilization_source_unknown"])
        )
        self.assertTrue(
            branch_render_is_usable(
                ["sleeptalk_called_unidentified",
                 "attract_immobilization_source_unknown"]
            )
        )

    def test_the_synthetic_empty_placeholder_is_never_usable(self) -> None:
        """It carries events ["|"], so it compares empty component lists on
        both slots and would MATCH any HP-free boundary against a branch that
        verified nothing."""

        self.assertFalse(branch_render_is_usable(["empty_instruction_list"]))
        self.assertFalse(
            branch_render_is_usable(
                ["sleeptalk_called_unidentified", "empty_instruction_list"]
            )
        )

    def test_an_unknown_future_marker_fails_CLOSED(self) -> None:
        """The point of the allowlist. An exclusion-shaped rule would admit
        this silently the moment the renderer grows a new mark_lossy call."""

        self.assertFalse(branch_render_is_usable(["some_future_marker"]))
        self.assertFalse(
            branch_render_is_usable(
                ["sleeptalk_called_unidentified", "some_future_marker"]
            )
        )


if __name__ == "__main__":
    unittest.main()
