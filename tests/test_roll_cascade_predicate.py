"""Pins for the roll-cascade fallback, written from the #1010 review findings.

The reviewer's counterexample is the first test: an observed Leftovers tick
against an engine Wish plus Leech Seed, nets equal. Different mechanics, and
the first version of the predicate accepted them.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("etd", ROOT / "scripts" / "engine_transition_differential.py")
D = importlib.util.module_from_spec(spec); sys.modules["etd"] = D; spec.loader.exec_module(D)


def comp(source: str, delta: int, index: int = 0):
    return D.DamageComponent(source=source, delta=delta, event_index=index)


def cascade(observed, engine, legal=None, observed_all=None, engine_all=None):
    """Exercise the REAL call path by default.

    Round eight: 22 of 25 tests omitted observed_all/engine_all, so they took
    the `is not None` fallback -- the roll-scaled-subset walk that round seven
    had just established was wrong. The pins were covering the code path the
    production caller does not use. Default them to the component lists, which
    is what evaluate_boundary_strict passes.
    """

    return D._roll_cascade_equivalent(
        observed,
        engine,
        support=None,
        target_side="side_one",
        pre_legal=legal,
        observed_all=observed_all if observed_all is not None else observed,
        engine_all=engine_all if engine_all is not None else engine,
    )


class RollCascadeTests(unittest.TestCase):
    def test_mismatched_heal_sources_are_rejected(self) -> None:
        """The review's counterexample. Nets agree at -75 on both sides, and the
        engine's heals are a different MECHANISM -- Wish plus Leech Seed against
        a Leftovers tick. Equal sums are not one transition at two rolls."""

        observed = [comp("", -100), comp("itemleftovers", 25)]
        engine = [comp("", -110), comp("movewish", 20), comp("leechseed", 15)]
        self.assertEqual(sum(c.delta for c in observed), sum(c.delta for c in engine))
        self.assertFalse(cascade(observed, engine))

    def test_the_genuine_cascade_is_accepted(self) -> None:
        """s18000588/37: one legal roll apart, the Wish stops capping, and a
        Leftovers residual fills the 4 HP it left."""

        observed = [comp("", -41), comp("movewish_to_full", 119)]
        engine = [comp("", -45), comp("movewish", 119), comp("itemleftovers_to_full", 4)]
        self.assertTrue(cascade(observed, engine))

    def test_the_direct_component_faces_the_ordinary_roll_check(self) -> None:
        """The first version returned before the per-component loop, so the
        direct damage never met ``roll_components_agree`` at all -- only a
        proportional band inside the cascade helper.

        Note what this does NOT assert. An empty ``legal`` set is not a veto:
        the module documents ``legal`` as "an ADDITIONAL accept path, never a
        veto", because it is computed from the pre-state with an assumed move
        order and letting it reject would fail boundaries where the two sims
        agree to the HP point. The ordinary path passes with ``legal=set()``
        too. Requiring a veto here would be inconsistent with the matcher.

        What the routing buys is that a direct component too far apart to be
        one roll is now rejected by the same rule every other roll-scaled
        component obeys."""

        heals_observed = [comp("movewish_to_full", 119)]
        heals_engine = [comp("movewish", 119), comp("itemleftovers_to_full", 4)]
        # one roll apart: accepted
        self.assertTrue(cascade([comp("", -41)] + heals_observed,
                                [comp("", -45)] + heals_engine))
        # far outside any roll, conservation still satisfied: rejected
        self.assertFalse(cascade([comp("", -10), comp("movewish_to_full", 119)],
                                 [comp("", -45), comp("movewish", 119),
                                  comp("itemleftovers_to_full", 35)]))

    def test_two_extra_components_are_rejected(self) -> None:
        observed = [comp("", -41), comp("movewish_to_full", 119)]
        engine = [comp("", -49), comp("movewish", 119), comp("itemleftovers", 4), comp("heal", 4)]
        self.assertFalse(cascade(observed, engine))

    def test_identical_directs_with_a_shifted_shared_heal_are_rejected(self) -> None:
        """Re-review hole (a). The direct components are IDENTICAL, so there is
        no roll gap and no cascade -- but comparing heal SOURCES only let the
        16 HP shift hide inside the 'extra' component."""

        self.assertFalse(cascade(
            [comp("", -100), comp("movewish", 60)],
            [comp("", -100), comp("movewish", 44), comp("itemleftovers", 16)],
        ))
        self.assertFalse(cascade(
            [comp("", -100), comp("itemleftovers", 25)],
            [comp("", -100), comp("itemleftovers", 9), comp("leechseed", 16)],
        ))

    def test_a_discrete_mechanic_difference_is_rejected(self) -> None:
        """Re-review hole (b). Conservation holds and the extra equals the roll
        gap, but no shared heal flipped its cap -- so nothing stopped topping
        the mon out and there is no cascade. The engine simply sees a Leech Seed
        the observation never did."""

        self.assertFalse(cascade(
            [comp("", -100), comp("itemleftovers", 25)],
            [comp("", -106), comp("itemleftovers", 25), comp("leechseed", 6)],
        ))

    def test_the_split_gap_majority_is_accepted(self) -> None:
        """C66's general identity: the roll gap equals the sum of increases
        across ALL healing components, not the extra alone.

        A previous round required the extra to EQUAL the gap. That is C66's
        narrow form, which the report measured at 21 rows and recorded as "NOT
        worth implementing", and it rejected the majority the report names --
        because in a real cascade the capped heal is precisely the one whose
        magnitude moved. These two are C66's own worked examples."""

        # s18000053/136: gap 6 = heal gap 3 + residual 3
        self.assertTrue(cascade(
            [comp("", -99), comp("movewish_to_full", 138)],
            [comp("", -105), comp("movewish", 141), comp("itemleftovers_to_full", 3)],
        ))
        # s18001200/15: gap 9 = heal gap 2 + residual 7
        self.assertTrue(cascade(
            [comp("", -113), comp("movewish_to_full", 113)],
            [comp("", -122), comp("movewish", 115), comp("itemleftovers_to_full", 7)],
        ))

    def test_the_extra_must_be_capped(self) -> None:
        """Both sides end at full in a cascade, so the residual that filled the
        gap tops the mon out. An untagged extra is a discrete mechanic the other
        side never saw."""

        self.assertFalse(cascade(
            [comp("", -41), comp("movewish_to_full", 119)],
            [comp("", -45), comp("movewish", 119), comp("itemleftovers", 4)],
        ))

    def test_the_cap_must_flip_on_the_smaller_roll(self) -> None:
        """The side with the smaller direct roll keeps more HP, so it is the one
        whose heal still caps. A flip the other way is impossible from a common
        start, and without the direction check it was admitted."""

        self.assertFalse(cascade(
            [comp("", -100), comp("movewish", 30), comp("itemleftovers", 25)],
            [comp("", -106), comp("movewish_to_full", 30), comp("itemleftovers", 25),
             comp("leechseed_to_full", 6)],
        ))

    def test_the_extra_may_not_duplicate_a_shared_source(self) -> None:
        """Otherwise the extra supplies its own cap flip and condition (3) is
        satisfied by the very component it is meant to constrain."""

        self.assertFalse(cascade(
            [comp("", -100), comp("itemleftovers", 25)],
            [comp("", -106), comp("itemleftovers", 25), comp("itemleftovers_to_full", 6)],
        ))

    def test_the_two_corpus_rows_a_previous_round_regressed(self) -> None:
        """Both are exact two-roll cascades, reproducible from a single common
        start, and a no-repeated-base rule rejected them. Two untagged -heal
        lines on one slot both normalise to `heal`; that is ordinary."""

        self.assertTrue(cascade(  # s18500122/85, gap 13 = heal gap 12 + extra 1
            [comp("", -149, 0), comp("heal", 132, 1), comp("heal_to_full", 17, 2)],
            [comp("", -162, 0), comp("heal", 132, 1), comp("heal", 29, 2),
             comp("itemleftovers_to_full", 1, 3)],
        ))
        self.assertTrue(cascade(  # s18100033/60, gap 2 = heal gap 0 + extra 2
            [comp("", -111, 0), comp("heal", 170, 1), comp("heal_to_full", 32, 2)],
            [comp("", -113, 0), comp("heal", 170, 1), comp("heal", 32, 2),
             comp("itemleftovers_to_full", 2, 3)],
        ))

    def test_cap_damage_cap_is_allowed(self) -> None:
        """Recover to full, take the hit, Leftovers tops out again. An earlier
        round banned two caps per side outright, which is physically false."""

        self.assertTrue(cascade(
            [comp("heal_to_full", 20, 0), comp("", -25, 1), comp("itemleftovers_to_full", 25, 2)],
            [comp("heal_to_full", 20, 0), comp("", -28, 1), comp("itemleftovers", 25, 2),
             comp("leechseed_to_full", 3, 3)],
        ))

    def test_two_caps_with_no_damage_between_are_rejected(self) -> None:
        """A heal that tops the mon out leaves nothing for a later heal, so a
        second positive _to_full needs damage in between. Order decides it."""

        self.assertFalse(cascade(
            [comp("", -99, 0), comp("movewish_to_full", 100, 1), comp("leechseed_to_full", 10, 2)],
            [comp("", -105, 0), comp("movewish", 103, 1), comp("leechseed_to_full", 10, 2),
             comp("itemleftovers_to_full", 3, 3)],
        ))

    def test_a_once_per_turn_residual_may_not_repeat(self) -> None:
        """Leftovers ticks once per turn, so a duplicate cannot be real -- one
        copy matched under instance pairing and the other supplied the flip."""

        self.assertFalse(cascade(
            [comp("", -99, 0), comp("itemleftovers_to_full", 10, 1), comp("itemleftovers", 10, 2)],
            [comp("", -105, 0), comp("itemleftovers", 13, 1), comp("itemleftovers", 10, 2),
             comp("movewish_to_full", 3, 3)],
        ))

    def test_the_extra_may_not_exceed_the_roll_gap(self) -> None:
        """A capped heal equals the deficit and the uncapped instance is the
        nominal that overshot it, so extra <= gap. Without this the predicate
        MASKED a real engine defect: a Leech Seed healing 21 where the simulator
        healed 25 from an identical state."""

        self.assertFalse(cascade(
            [comp("", -64), comp("movewish", 118), comp("leechseed_to_full", 25)],
            [comp("", -65), comp("movewish", 118), comp("leechseed", 21),
             comp("itemleftovers_to_full", 5)],
        ))

    def test_the_capped_instance_may_not_exceed_its_nominal(self) -> None:
        self.assertFalse(cascade(
            [comp("", -99), comp("movewish_to_full", 150)],
            [comp("", -105), comp("movewish", 141), comp("itemleftovers_to_full", 15)],
        ))

    def test_unequal_totals_are_rejected(self) -> None:
        observed = [comp("", -41), comp("movewish_to_full", 119)]
        engine = [comp("", -45), comp("movewish", 119), comp("itemleftovers_to_full", 40)]
        self.assertFalse(cascade(observed, engine))

    def test_a_negative_heal_is_rejected(self) -> None:
        """An impossible component must never license a match (C52)."""

        observed = [comp("", -41), comp("movewish_to_full", 119)]
        engine = [comp("", -45), comp("movewish", 119), comp("itemleftovers", -4)]
        self.assertFalse(cascade(observed, engine))


if __name__ == "__main__":
    unittest.main()


class PhysicalFloorTests(unittest.TestCase):
    """Pins for the two constraints added after review round six.

    Round six built an independent physical-realizability oracle and found two
    shapes the predicate accepted that cannot happen. Both are pinned here, and
    both pins were mutation-checked: deleting the constraint makes the matching
    test fail.
    """

    def test_the_extra_must_sit_on_the_larger_roll_side(self) -> None:
        """Conservation would force the cap BELOW its physical floor.

        At event 0 both sims are at the same HP, so a heal cannot top out at 100
        on one side while paying 120 on the other. capped = uncapped - gap -
        extra = 100, but the floor is uncapped - gap = 110.
        """

        observed = [comp("movewish", 120, 0), comp("", -210, 1)]
        engine = [
            comp("movewish_to_full", 100, 0),
            comp("", -200, 1),
            comp("itemleftovers_to_full", 10, 2),
        ]
        self.assertEqual(sum(c.delta for c in observed), sum(c.delta for c in engine))
        self.assertFalse(cascade(observed, engine))

    def test_a_cap_may_not_be_smaller_than_the_damage_before_it(self) -> None:
        """A `_to_full` restores maxhp - hp_before, so it is bounded BELOW by the
        damage already taken. 150 after a -200 is impossible, not merely small.

        This is the residual the PR body called out and said needed component
        ordering the renderer does not emit. event_index already carries it.
        """

        observed = [comp("", -200, 0), comp("movewish_to_full", 150, 1)]
        engine = [
            comp("", -212, 0),
            comp("movewish", 160, 1),
            comp("itemleftovers_to_full", 2, 2),
        ]
        self.assertEqual(sum(c.delta for c in observed), sum(c.delta for c in engine))
        self.assertFalse(cascade(observed, engine))

    def test_a_cap_flip_before_the_only_damage_is_rejected(self) -> None:
        """Constraint 4 orders caps against damage only when a side has TWO. With
        a single cap there was no floor, so a flip could sit before the damage
        that supposedly caused it."""

        observed = [comp("movewish", 120, 0), comp("", -210, 1), comp("heal", 250, 2)]
        engine = [
            comp("movewish_to_full", 100, 0),
            comp("", -200, 1),
            comp("heal", 250, 2),
            comp("itemleftovers_to_full", 10, 3),
        ]
        self.assertFalse(cascade(observed, engine))

    def test_a_genuine_repeated_heal_cascade_still_passes(self) -> None:
        """The guard against over-tightening. hp0 = maxhp, base 112, rolls
        100/106: two untagged heals on one slot is ordinary, and this pair IS
        realizable. An earlier round banned repeated bases outright and
        regressed two real corpus rows."""

        observed = [comp("", -100, 0), comp("heal", 40, 1), comp("heal_to_full", 60, 2)]
        engine = [
            comp("", -106, 0),
            comp("heal", 40, 1),
            comp("heal", 60, 2),
            comp("itemleftovers_to_full", 6, 3),
        ]
        self.assertTrue(cascade(observed, engine))


class PrefixWalkScopeTests(unittest.TestCase):
    """The prefix floor must walk ALL HP events, not the roll-scaled subset.

    Round seven: the first version of the floor summed only the components that
    reach the predicate (obs_rolled / eng_rolled). Exact-bucket components --
    an untagged Leftovers tick, sethp, burn/poison/sandstorm -- move HP but are
    not in that list, so `running` was not the real running total. It
    false-rejected a realizable cascade, the same over-tightening class that
    regressed two corpus rows in an earlier round.
    """

    # maxhp 300, start 290/300, damage base 155 (rolls 94 -> 145, 100 -> 155),
    # Leftovers floor(300/16) = 18, Wish nominal 140, Leech Seed 30. Both sides
    # end at exactly 300/300.
    OBS_ALL = [
        comp("", -145, 0),
        comp("itemleftovers", 18, 1),
        comp("movewish_to_full", 137, 2),
    ]
    ENG_ALL = [
        comp("", -155, 0),
        comp("itemleftovers", 18, 1),
        comp("movewish", 140, 2),
        comp("leechseed_to_full", 7, 3),
    ]
    OBS_ROLLED = [comp("", -145, 0), comp("movewish_to_full", 137, 2)]
    ENG_ROLLED = [
        comp("", -155, 0),
        comp("movewish", 140, 2),
        comp("leechseed_to_full", 7, 3),
    ]

    def test_an_exact_bucket_heal_is_counted_in_the_running_total(self) -> None:
        """The +18 Leftovers makes the true deficit at the cap 137, not 145."""

        self.assertTrue(
            D._roll_cascade_equivalent(
                self.OBS_ROLLED,
                self.ENG_ROLLED,
                support=None,
                target_side="side_one",
                pre_legal=None,
                observed_all=self.OBS_ALL,
                engine_all=self.ENG_ALL,
            )
        )

    def test_the_subset_walk_is_what_got_this_wrong(self) -> None:
        """Guard against silently dropping the threading: without the full
        lists the same pair is rejected, so a caller that forgets to pass them
        is measurably more conservative rather than accidentally identical."""

        self.assertFalse(
            D._roll_cascade_equivalent(
                self.OBS_ROLLED,
                self.ENG_ROLLED,
                support=None,
                target_side="side_one",
                pre_legal=None,
            )
        )

    def test_the_floor_still_rejects_an_impossible_cap_with_full_lists(self) -> None:
        """Widening the walk must not cost soundness."""

        observed = [comp("", -200, 0), comp("movewish_to_full", 150, 1)]
        engine = [
            comp("", -212, 0),
            comp("movewish", 160, 1),
            comp("itemleftovers_to_full", 2, 2),
        ]
        self.assertFalse(
            D._roll_cascade_equivalent(
                observed,
                engine,
                support=None,
                target_side="side_one",
                pre_legal=None,
                observed_all=observed,
                engine_all=engine,
            )
        )


class PreDirectHitTests(unittest.TestCase):
    """Nothing may differ before the roll that caused the difference.

    Round eight's finding, and the last unsound accept class the review's
    oracle could find. Two arms of one cascade share a start; the direct hit is
    the only roll-dependent event, so below its event_index both arms sit at
    identical HP and identical deficit. A cap cannot flip there and an extra
    cannot appear there.
    """

    def test_a_cap_flip_before_the_direct_hit_is_rejected(self) -> None:
        observed = [comp("movewish_to_full", 100, 0), comp("", -102, 2)]
        engine = [
            comp("movewish", 104, 0),
            comp("itemleftovers_to_full", 4, 1),
            comp("", -110, 2),
        ]
        self.assertFalse(cascade(observed, engine))

    def test_an_extra_before_the_direct_hit_is_rejected(self) -> None:
        """From the review's unbiased 181k sweep -- one of only two accepts,
        and its oracle called both unrealizable."""

        observed = [comp("movewish", 49, 0), comp("heal_to_full", 1, 1), comp("", -37, 2)]
        engine = [comp("movewish_to_full", 43, 0), comp("", -30, 2)]
        self.assertFalse(cascade(observed, engine))

    def test_the_other_sweep_accept_is_rejected(self) -> None:
        observed = [
            comp("leechseed", 44, 0),
            comp("", -24, 1),
            comp("itemleftovers_to_full", 4, 2),
        ]
        engine = [comp("leechseed_to_full", 43, 0), comp("", -19, 1)]
        self.assertFalse(cascade(observed, engine))

    def test_agreeing_components_before_the_direct_hit_are_fine(self) -> None:
        """The guard must reject DISAGREEMENT, not the mere presence of an
        earlier component. Both arms carry the same Leftovers tick at @0."""

        observed = [comp("itemleftovers", 6, 0), comp("", -100, 1), comp("heal_to_full", 94, 2)]
        engine = [
            comp("itemleftovers", 6, 0),
            comp("", -106, 1),
            comp("heal", 94, 2),
            comp("leechseed_to_full", 6, 3),
        ]
        self.assertTrue(cascade(observed, engine))
