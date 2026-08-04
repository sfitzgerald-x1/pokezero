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
    damage_components,
    _ROLL_SCALED_SOURCES,
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

    def test_the_cascade_path_reaches_the_comparator(self) -> None:
        """Coverage for the length-mismatch path.

        NOT a threading pin, despite what this used to claim. Review: the
        cascade site only ever compares the DIRECT components, whose source is
        `""` -- never `_to_full`, always roll-scaled -- so the cap tolerance and
        therefore `damage_scales` are unreachable there. The threading argument
        was dead code and has been removed; this test passed with or without it.
        The real threading pin is the sibling test above.
        """

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



class PainSplitIsRollAware(unittest.TestCase):
    """`movepainsplit` inherits the damage roll and must not be EXACT-bucketed.

    Pain Split sets both mons to ``floor((hp_a + hp_b) / 2)``, so its magnitude
    is a function of the HP left after whatever damage landed earlier in the same
    turn. Treating it as deterministic was a matcher defect; it produced the whole
    `I3_roll_inherited` family (reports/c95, reports/c101).

    Reverting the `movepainsplit` entry in `_ROLL_SCALED_SOURCES` makes every case
    below fail, which is the point: these are the four rows the sweep closed
    (19000016/2, 19000016/49, 19000071/55, 19000198/22), reduced to their
    deciding arms.
    """

    def test_membership(self) -> None:
        self.assertIn("movepainsplit", _ROLL_SCALED_SOURCES)

    def test_the_four_closed_rows_agree(self) -> None:
        # (observed, engine) from the deciding arm of each closed row.
        for observed, engine, row in (
            (-5, -4, "19000016/2"),
            (53, 52, "19000016/49"),
            (24, 23, "19000071/55"),
            (-91, -89, "19000198/22 (6.25% crit arm)"),
        ):
            with self.subTest(row=row):
                self.assertTrue(
                    roll_components_agree(
                        [("movepainsplit", observed)],
                        [("movepainsplit", engine)],
                        None,
                    ),
                    f"{row}: {observed} vs {engine} must agree once roll-aware",
                )

    def test_the_small_magnitude_case_needs_the_absolute_slack(self) -> None:
        """19000016/2 is why the band needs its +/-1, not just its ratio.

        A delta of 1 on a magnitude of 5 is 20%, far outside the 0.92-1.09
        ratio. It passes only because the window carries one HP of flooring
        slack in both directions. An earlier report claimed this row could not be
        closed by roll-awareness, having read the ratio from a docstring instead
        of the predicate.
        """
        low = abs(-4) * 0.92 - 1
        high = abs(-4) * 1.09 + 1
        self.assertLessEqual(low, 5)
        self.assertLessEqual(5, high)
        self.assertGreater(5, abs(-4) * 1.09, "ratio alone would reject this")

    def test_a_genuinely_different_arm_is_still_rejected(self) -> None:
        """19000198/22's 93.75% arm is non-crit and legitimately differs.

        Roll-awareness must not turn into blanket acceptance: -91 against -120 is
        outside the window and must stay a mismatch, so the fix still discriminates
        per arm across the branch set.
        """
        self.assertFalse(
            roll_components_agree(
                [("movepainsplit", -91)], [("movepainsplit", -120)], None
            )
        )



class FaintWithoutADamageEventIsSynthesised(unittest.TestCase):
    """Destiny Bond and Perish Song kill without emitting a `-damage` line.

    Showdown announces them with `|-activate|` or `|-start| perish0` plus
    `|faint|`. The engine models the same state change as a Damage instruction
    for the victim's whole remaining HP, so before this the observation carried
    NOTHING for that slot while the engine carried a `capped_lethal`, and the
    comparison had no counterpart. Six rows (reports/c96, c103).

    Reverting the `tag == "faint"` arm in the parser makes the first two cases
    return empty lists, which is the defect.
    """

    def test_destiny_bond_faint_is_synthesised(self) -> None:
        got = damage_components(
            [
                "|move|p2a: Qwilfish|Destiny Bond|p2a: Qwilfish",
                "|move|p1a: Clefable|Return|p2a: Qwilfish",
                "|-damage|p2a: Qwilfish|0 fnt",
                "|faint|p2a: Qwilfish",
                "|-activate|p2a: Qwilfish|move: Destiny Bond",
                "|faint|p1a: Clefable",
            ],
            {"p1": 95, "p2": 88},
        )
        # p1 has no -damage line at all; its faint must still be a component.
        self.assertEqual(got["p1"], [("capped_lethal", -95)])
        self.assertEqual(got["p2"], [("capped_lethal", -88)])

    def test_perish_song_faint_is_synthesised(self) -> None:
        got = damage_components(
            [
                "|move|p1a: Misdreavus|Mean Look|p2a: Suicune",
                "|cant|p2a: Suicune|slp",
                "|-start|p2a: Suicune|perish0",
                "|upkeep",
                "|faint|p2a: Suicune",
            ],
            {"p1": 241, "p2": 270},
        )
        self.assertEqual(got["p2"], [("capped_lethal", -270)])
        self.assertEqual(got["p1"], [])

    def test_an_ordinary_faint_is_not_double_counted(self) -> None:
        """The load-bearing control.

        A normal KO already emits `-damage ... 0 fnt`, so `running[slot]` is
        zero by the time `|faint|` arrives and the synthesis must be a no-op.
        Without that guard every KO in the corpus would gain a phantom second
        component.
        """
        got = damage_components(
            ["|move|p1a: A|Tackle|p2a: B", "|-damage|p2a: B|0 fnt", "|faint|p2a: B"],
            {"p1": 100, "p2": 50},
        )
        self.assertEqual(got["p2"], [("capped_lethal", -50)])

    def test_a_faint_on_an_untracked_slot_is_ignored(self) -> None:
        """No initial HP for the slot means nothing to synthesise from."""
        got = damage_components(["|faint|p2a: B"], {"p1": 100})
        self.assertEqual(got["p2"], [])


if __name__ == "__main__":
    unittest.main()


class CappedLethalCascade(unittest.TestCase):
    """Both sides clipped: the cap difference IS the direct-damage difference.

    reports/c88 sub-shape A. A residual that kills is clipped to whatever HP
    remains, so each arm clips to its OWN remainder and the arm that took LESS
    direct damage has the LARGER cap. Conservation is exact. The old rule
    asserted the observed cap "can only ever be SMALLER than the uncapped tick
    the engine carries" -- sound when only ONE side is capped, false when both
    are -- and rejected pairs that conserve to the point.

    The mirror of the capped-HEAL cascade, in the damage direction.
    """

    def test_the_real_corpus_rows_now_match(self) -> None:
        for ident, obs_direct, obs_cap, eng_direct, eng_cap in (
            ("19000002/61", -125, -40, -133, -32),
            ("19000016/85", -45, -67, -47, -65),
        ):
            with self.subTest(row=ident):
                self.assertEqual(obs_direct + obs_cap, eng_direct + eng_cap)
                self.assertTrue(
                    roll_component_events_agree(
                        [comp("", obs_direct, 0), comp("capped_lethal", obs_cap, 1)],
                        [comp("", eng_direct, 0), comp("capped_lethal", eng_cap, 1)],
                        support=None, target_side="side_one", pre_legal=None,
                    )
                )

    def test_the_arm_that_took_more_damage_must_have_the_smaller_cap(self) -> None:
        """Direction. Inverting it is physically impossible: less HP left
        cannot produce a larger clip."""

        self.assertFalse(
            roll_component_events_agree(
                [comp("", -133, 0), comp("capped_lethal", -40, 1)],
                [comp("", -125, 0), comp("capped_lethal", -32, 1)],
                support=None, target_side="side_one", pre_legal=None,
            )
        )

    def test_the_mirror_direction_is_also_rejected(self) -> None:
        """Review found the eng_direct > obs_direct guard UNPINNED: deleting it
        passed all 103 tests. The pinning test asserted only one half while its
        docstring claimed both. The equality form subsumes both guards, and this
        pins the half that was uncovered."""

        self.assertFalse(
            roll_component_events_agree(
                [comp("", -125, 0), comp("capped_lethal", -32, 1)],
                [comp("", -133, 0), comp("capped_lethal", -40, 1)],
                support=None, target_side="side_one", pre_legal=None,
            )
        )

    def test_the_one_hp_slack_cannot_be_abused(self) -> None:
        """The old +/-1 slack let the nets differ by 1 HP in a step whose value
        is exactly determined -- implying two different pre-HP values for one
        boundary. The equality rejects it."""

        self.assertFalse(
            roll_component_events_agree(
                [comp("", -125, 0), comp("capped_lethal", -41, 1)],
                [comp("", -133, 0), comp("capped_lethal", -32, 1)],
                support=None, target_side="side_one", pre_legal=None,
            )
        )

    def test_a_cap_gap_wider_than_the_direct_gap_is_rejected(self) -> None:
        self.assertFalse(
            roll_component_events_agree(
                [comp("", -125, 0), comp("capped_lethal", -60, 1)],
                [comp("", -133, 0), comp("capped_lethal", -32, 1)],
                support=None, target_side="side_one", pre_legal=None,
            )
        )

    def test_the_basis_excludes_the_cap_itself(self) -> None:
        """The trap. Conservation makes the SLOT-WIDE totals identical -- 165
        vs 165, difference 0 -- so a slot-wide bound rejects every one of these.
        Passing the slot basis where the direct basis belongs reproduces that.
        """

        self.assertFalse(
            roll_components_agree(
                [("capped_lethal", -40)], [("capped_lethal", -32)], None,
                damage_scales=(165, 165), direct_damage_scales=(165, 165),
            )
        )
        self.assertTrue(
            roll_components_agree(
                [("capped_lethal", -40)], [("capped_lethal", -32)], None,
                damage_scales=(165, 165), direct_damage_scales=(125, 133),
            )
        )
