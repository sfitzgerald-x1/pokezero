"""Unit tests for the strict transition matcher's component comparator.

The comparator decides every divergence verdict in the acceptance measurement,
so its tolerances are pinned here directly rather than only through end-to-end
census numbers. A lenient comparator produces clean aggregates, which is exactly
the failure mode that looks like success.
"""

from __future__ import annotations

from collections import Counter
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from engine_transition_differential import (  # noqa: E402
    _split_components,
    classify_divergence,
    count_world_construction_limit,
    damage_components,
    roll_components_agree,
    world_construction_limit,
)
from pokezero.engine_world import EngineWorldUnsupported  # noqa: E402


class WorldConstructionLimits(unittest.TestCase):
    def test_unknown_substitute_health_is_a_named_limit(self):
        error = EngineWorldUnsupported("substitute_health_unknown", "public hit without amount")
        self.assertEqual(
            world_construction_limit(error),
            "limit:world_substitute_health_unknown",
        )

    def test_other_world_errors_remain_non_limit_skips(self):
        error = EngineWorldUnsupported("payload_malformed", "bad payload")
        self.assertIsNone(world_construction_limit(error))

    def test_named_limit_accounting_is_reusable_across_constructor_passes(self):
        counts = Counter()
        error = EngineWorldUnsupported("substitute_health_unknown", "public hit without amount")
        self.assertTrue(count_world_construction_limit(counts, error))
        self.assertTrue(count_world_construction_limit(counts, error))
        self.assertEqual(counts["limit:world_substitute_health_unknown"], 2)
        self.assertNotIn("skip:world_unsupported:substitute_health_unknown", counts)

    def test_non_limit_is_not_counted_by_limit_accounting(self):
        counts = Counter()
        error = EngineWorldUnsupported("payload_malformed", "bad payload")
        self.assertFalse(count_world_construction_limit(counts, error))
        self.assertEqual(counts, Counter())

    def test_substitute_provenance_contradiction_is_never_a_limit(self):
        counts = Counter()
        error = EngineWorldUnsupported(
            "substitute_health_provenance_contradiction",
            "active Substitute has missing provenance",
        )
        self.assertIsNone(world_construction_limit(error))
        self.assertFalse(count_world_construction_limit(counts, error))
        self.assertEqual(counts, Counter())


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


class SleepTalkUnknownCallee(unittest.TestCase):
    """A Sleep Talk branch the mapper could not attribute is still validatable.

    The mapper cannot recover WHICH move Sleep Talk called, so it renders the
    called move's damage as `[from] residual` and flags the branch. That routed
    real move damage into the EXACT bucket, where it could never match
    Showdown's bare `-damage` line. The engine still branches over the candidate
    calls and the matching branch is present — only the label is missing — so
    the damage is reclassified as roll-scaled and matched against the union.

    Both pins come from rows that were replayed by hand:
      seed 1350014 step 55 — Sleep Talk called Seismic Toss for exactly -78
      seed 1350019 step 99 — called Psychic for -97 vs Showdown's -103
    """

    ENGINE_LINE = "|-damage|p1a: Armaldo|122/260 par|[from] residual"

    def test_unattributed_damage_stays_exact_by_default(self):
        got = damage_components(
            [self.ENGINE_LINE], {"p1": 200}, unattributed_damage_as_roll=False
        )
        exact, rolled = _split_components(got["p1"])
        self.assertEqual(sorted(exact.elements()), [("residual", -78)])
        self.assertEqual(rolled, [])

    def test_unattributed_damage_becomes_roll_scaled_when_flagged(self):
        got = damage_components(
            [self.ENGINE_LINE], {"p1": 200}, unattributed_damage_as_roll=True
        )
        exact, rolled = _split_components(got["p1"])
        self.assertEqual(list(exact.elements()), [])
        self.assertEqual(rolled, [("move_unknown_callee", -78)])

    def test_seed_1350014_step_55_exact_damage_PASSES(self):
        """The replayed row: Showdown -78 bare, engine -78 unattributed."""

        observed = damage_components(
            ["|-damage|p1a: Armaldo|122/260 par"], {"p1": 200}
        )["p1"]
        engine = damage_components(
            [self.ENGINE_LINE], {"p1": 200}, unattributed_damage_as_roll=True
        )["p1"]
        self.assertTrue(roll_components_agree(
            _split_components(observed)[1], _split_components(engine)[1], None))

    def test_seed_1350019_step_99_in_window_damage_PASSES(self):
        """Showdown -103 against the engine's -97: inside the roll window."""

        self.assertTrue(roll_components_agree(
            [("", -103)], [("move_unknown_callee", -97)], None))

    def test_fabricated_wrong_damage_FAILS(self):
        """The guard: a callee whose damage is nowhere near must still diverge."""

        self.assertFalse(roll_components_agree(
            [("", -78)], [("move_unknown_callee", -20)], None))
        self.assertFalse(roll_components_agree(
            [("", -78)], [("move_unknown_callee", -200)], None))

    def test_named_residual_is_NOT_reclassified(self):
        """The containment boundary, pinned.

        The predicate keys on the mapper's generic `[from] residual` fallback,
        NOT on "the called move's damage" — nothing in the rendered stream says
        which line is the callee's. So a residual the mapper DID attribute must
        keep its exact comparison even inside a Sleep-Talk-flagged branch. The
        census shows this empirically; without this pin nothing enforces it.
        """

        lines = [
            "|-damage|p1a: A|100/260|[from] psn",
            "|-damage|p1a: A|60/260|[from] Sandstorm",
            "|-damage|p1a: A|20/260|[from] residual",
        ]
        got = damage_components(lines, {"p1": 130}, unattributed_damage_as_roll=True)
        exact, rolled = _split_components(got["p1"])
        # Named residuals stay EXACT ...
        self.assertEqual(
            sorted(exact.elements()), [("psn", -30), ("sandstorm", -40)]
        )
        # ... only the unattributed one is reclassified.
        self.assertEqual(rolled, [("move_unknown_callee", -40)])

    def test_unattributed_HEAL_is_not_reclassified(self):
        """Reclassification is scoped to -damage; heals keep their own rules."""

        got = damage_components(
            ["|-heal|p1a: A|150/260|[from] residual"],
            {"p1": 130},
            unattributed_damage_as_roll=True,
        )
        exact, rolled = _split_components(got["p1"])
        self.assertEqual(sorted(exact.elements()), [("residual", 20)])
        self.assertEqual(rolled, [])

    def test_missing_damage_entirely_FAILS(self):
        """Reclassification must not make an absent component match a present one."""

        self.assertFalse(roll_components_agree([("", -78)], [], None))


class LengthMismatch(unittest.TestCase):
    def test_differing_component_counts_never_agree(self):
        self.assertFalse(roll_components_agree([("", -50)], [], None))
        self.assertFalse(roll_components_agree([], [("", -50)], None))


if __name__ == "__main__":
    unittest.main()


class PainSplitSetHp(unittest.TestCase):
    """`|-sethp|` must be consumed, or Pain Split corrupts the NEXT component.

    Showdown expresses Pain Split as two `-sethp` lines — the target's first and
    `[silent]`, then the user's — and it is the only move in the gen3 randbats
    pool that emits the tag (reachable on dusclops, misdreavus, swalot,
    weezing). While the extractor accepted only `-damage`/`-heal`, the HP change
    was dropped and its magnitude was absorbed into the next attributed delta on
    that slot, so the instrument manufactured impossible components and charged
    them to the engine.
    """

    # The verbatim Showdown slice from seed 1500008 step 101, the row that
    # exposed the gap. Dusclops (p1) at 132/209 Pain Splits Wigglytuff (p2) at
    # 125/407; both land on 128, then both tick Leftovers.
    SLICE = [
        "|move|p2a: Wigglytuff|Double-Edge|p1a: Dusclops",
        "|-immune|p1a: Dusclops",
        "|move|p1a: Dusclops|Pain Split|p2a: Wigglytuff",
        "|-sethp|p2a: Wigglytuff|128/407|[from] move: Pain Split|[silent]",
        "|-sethp|p1a: Dusclops|128/209|[from] move: Pain Split",
        "|-heal|p2a: Wigglytuff|153/407|[from] item: Leftovers",
        "|-heal|p1a: Dusclops|141/209|[from] item: Leftovers",
    ]

    def test_end_to_end_pin_seed_1500008_step_101(self):
        got = damage_components(self.SLICE, {"p1": 132, "p2": 125})
        # Pain Split is deterministic: floor((132 + 125) / 2) = 128 for both.
        self.assertEqual(got["p1"], [("movepainsplit", -4), ("itemleftovers", 13)])
        self.assertEqual(got["p2"], [("movepainsplit", 3), ("itemleftovers", 25)])

    def test_the_leftovers_ticks_are_the_TRUE_amounts(self):
        """The regression this fixes, stated as the number that was wrong.

        The engine emitted +13/+25 and was called divergent for it. Before the
        fix the harness reported +9/+28 — the deltas from the PRE-STEP HP, with
        Pain Split's -4/+3 silently folded in.
        """
        got = damage_components(self.SLICE, {"p1": 132, "p2": 125})
        lefties = {
            slot: [d for src, d in got[slot] if src == "itemleftovers"]
            for slot in ("p1", "p2")
        }
        self.assertEqual(lefties, {"p1": [13], "p2": [25]})
        self.assertNotIn(9, lefties["p1"])
        self.assertNotIn(28, lefties["p2"])

    def test_the_silent_target_line_is_NOT_skipped(self):
        """Showdown marks the target's half `[silent]`; it still moves HP."""
        got = damage_components(self.SLICE, {"p1": 132, "p2": 125})
        self.assertIn(("movepainsplit", 3), got["p2"])

    def test_pain_split_is_compared_EXACTLY_not_roll_scaled(self):
        exact, rolled = _split_components([("movepainsplit", -4)])
        self.assertEqual(dict(exact), {("movepainsplit", -4): 1})
        self.assertEqual(rolled, [])

    def test_a_four_point_disagreement_on_pain_split_FAILS(self):
        """Being deterministic, Pain Split gets no roll tolerance at all."""
        self.assertFalse(
            roll_components_agree(
                [("movepainsplit", -4)], [("movepainsplit", -8)], None
            )
        )

    def test_untagged_sethp_does_not_fall_into_the_roll_scaled_bucket(self):
        got = damage_components(
            ["|-sethp|p1a: Dusclops|128/209"], {"p1": 132}
        )
        self.assertEqual(got["p1"], [("sethp", -4)])
        exact, rolled = _split_components(got["p1"])
        self.assertEqual(rolled, [])


class MajorityBranchOverride(unittest.TestCase):
    """#946's adjudication, made mechanical.

    `branch_misses` is in branch order, so `misses[0]` may be a MINORITY branch.
    A row whose 6.25% branch lacks a Leftovers tick, while the 93.75% of
    probability mass complains only that the damage disagrees, is a damage
    disagreement — not a Leftovers one.
    """

    # s1500014 st69, verbatim. The row carried
    # `component_missing_in_engine:itemleftovers` for four cycles.
    PINNED = [
        "pct=6.25: p1 attributed components differ: "
        "observed_only=[('itemleftovers', 18)] engine_only=[]",
        "pct=75.00: p2 roll-scaled components differ: "
        "observed=[('', -214)] engine=[('', -116)]",
        "pct=18.75: p2 roll-scaled components differ: "
        "observed=[('', -214)] engine=[('', -116)]",
    ]

    def test_pinned_row_relabels_to_the_damage_class(self):
        self.assertEqual(classify_divergence([], self.PINNED), "roll_scaled_component")

    def test_a_genuine_residual_miss_is_NOT_overridden(self):
        """When the majority branch also blames the residual, the label stands."""
        misses = [
            "pct=25.00: p1 attributed components differ: "
            "observed_only=[('itemleftovers', 18)] engine_only=[]",
            "pct=75.00: p1 attributed components differ: "
            "observed_only=[('itemleftovers', 18)] engine_only=[]",
        ]
        self.assertEqual(
            classify_divergence([], misses),
            "component_missing_in_engine:itemleftovers",
        )

    def test_the_override_never_moves_a_row_into_a_LIMIT_class(self):
        """A relabel must not reduce the residue.

        A `capped_lethal` majority classifies as `limit:roll_divergent_lethality`
        — an adjudicated NON-divergence. Allowing the override there would move
        rows out of the outside-limit count and hand the acceptance gate a credit
        nobody adjudicated.
        """
        misses = [
            "pct=6.25: p1 attributed components differ: "
            "observed_only=[('itemleftovers', 18)] engine_only=[]",
            "pct=93.75: p2 roll-scaled components differ: "
            "observed=[('capped_lethal', -14)] engine=[('', -116)]",
        ]
        self.assertEqual(
            classify_divergence([], misses),
            "component_missing_in_engine:itemleftovers",
        )

    def test_an_unattributed_majority_does_not_trigger_the_override(self):
        """Only NAMED residuals are adjudicable; '' is not 'the residual'."""
        misses = [
            "pct=10.00: p1 attributed components differ: "
            "observed_only=[('abilityroughskin', 18)] engine_only=[]",
            "pct=90.00: p2 roll-scaled components differ: "
            "observed=[('', -214)] engine=[('', -116)]",
        ]
        self.assertEqual(
            classify_divergence([], misses),
            "component_missing_in_engine:abilityroughskin",
        )


class PairBySource(unittest.TestCase):
    """Components pair by SOURCE; magnitude is only a tiebreak within a source."""

    def test_a_residual_is_not_paired_against_a_move_hit(self):
        from triage_roll_components import _pair_by_source

        # The cycle-seven defect: sorted by bare magnitude, the 2-point residual
        # pairs with the 136-point move hit and yields a ratio of 0.015.
        observed = [("", -139), ("recoil", -2)]
        engine = [("", -136), ("recoil", -2)]
        paired = _pair_by_source(observed, engine)
        self.assertIn(("", 139, 136), paired)
        self.assertIn(("recoil", 2, 2), paired)
        self.assertNotIn(("", 2, 136), paired)

    def test_an_unmatched_component_yields_no_ratio(self):
        """A count difference is structural; it must not become a ratio."""
        from triage_roll_components import _pair_by_source

        self.assertEqual(_pair_by_source([("drain", -10)], []), [])
