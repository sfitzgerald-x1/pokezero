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
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import engine_transition_differential as differential  # noqa: E402
from engine_transition_differential import (  # noqa: E402
    _split_components,
    branch_event_legal_rolls,
    branch_component_legal_rolls,
    classify_divergence,
    count_world_construction_limit,
    damage_components,
    roll_component_events_agree,
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

    def test_substitute_sample_incompatibility_is_never_a_limit(self):
        counts = Counter()
        error = EngineWorldUnsupported(
            "substitute_depletion_world_incompatible",
            "sampled Substitute could not have survived exact depletion",
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


class EventAwareBranchLegality(unittest.TestCase):
    """Legal roll support must follow state changes that precede a hit."""

    def setUp(self):
        self._original_poke_engine = differential.poke_engine
        self.loaded_states: list[str] = []

        def from_string(value: str) -> str:
            self.loaded_states.append(value)
            return value

        def calculate_damage(state: str, _side_one: str, _side_two: str, _critical: bool):
            # 25 has the Gen 3 85..100% support 21..25. The retained Calm Mind
            # row needs 21 after the boost; the stale pre-state range was 24..29.
            self.assertIn(state, {"post-boost", "post-switch", "after-drop"})
            if state == "after-drop":
                # Aurora Beam is the earlier p1 -> p2 hit. The affected later
                # Tackle targets p1, so only side_two's rolls are valid there.
                return [90], [25]
            return [25], []

        differential.poke_engine = SimpleNamespace(
            State=SimpleNamespace(from_string=from_string),
            calculate_damage=calculate_damage,
        )

    def tearDown(self):
        differential.poke_engine = self._original_poke_engine

    def test_uses_pre_damage_snapshot_after_a_same_turn_stat_boost(self):
        legal = branch_event_legal_rolls(
            {
                "events": [
                    "|move|p2a: Clefable|Calm Mind|p2a: Clefable",
                    "|-boost|p2a: Clefable|spd|1",
                    "|move|p1a: Clefable|Fire Blast|p2a: Clefable",
                    "|-damage|p2a: Clefable|279/300 par",
                ],
                "legal_roll_state": "post-boost",
            },
            side_one_choice="fireblast",
            side_two_choice="calmmind",
        )

        self.assertEqual(legal.target_side, "p2")
        self.assertEqual(legal.damages, {21, 22, 23, 24, 25})
        self.assertEqual(self.loaded_states, ["post-boost"])

    def test_uses_pre_damage_snapshot_after_a_same_turn_switch(self):
        legal = branch_event_legal_rolls(
            {
                "events": [
                    "|switch|p2a: Lanturn|Lanturn, L82, M|143/339 tox",
                    "|move|p1a: Sableye|knockoff|p2a: Lanturn",
                    "|-damage|p2a: Lanturn|122/339 tox",
                ],
                "legal_roll_state": "post-switch",
            },
            side_one_choice="knockoff",
            side_two_choice="lanturn",
        )

        self.assertEqual(legal.target_side, "p2")
        self.assertEqual(legal.damages, {21, 22, 23, 24, 25})
        self.assertEqual(self.loaded_states, ["post-switch"])

    def test_ignores_state_changes_that_follow_direct_damage(self):
        legal = branch_event_legal_rolls(
            {
                "events": [
                    "|move|p1a: A|Flamethrower|p2a: B",
                    "|-damage|p2a: B|100/200",
                    "|-boost|p1a: A|spa|1",
                ],
                "post_state": "must-not-load",
            },
            side_one_choice="flamethrower",
            side_two_choice="splash",
        )

        self.assertIsNone(legal)
        self.assertEqual(self.loaded_states, [])

    def test_ignores_a_self_hp_cost_after_a_same_turn_stat_boost(self):
        legal = branch_event_legal_rolls(
            {
                "events": [
                    "|move|p1a: Rattata|Calm Mind|p1a: Rattata",
                    "|-boost|p1a: Rattata|spa|1",
                    "|move|p2a: Chansey|Substitute|p2a: Chansey",
                    "|-damage|p2a: Chansey|75/100",
                ],
            },
            side_one_choice="calmmind",
            side_two_choice="substitute",
        )

        self.assertIsNone(legal)
        self.assertEqual(self.loaded_states, [])

    def test_ignores_a_capped_stat_event_before_an_opponent_hit(self):
        legal = branch_event_legal_rolls(
            {
                "events": [
                    "|move|p1a: Rattata|Swords Dance|p1a: Rattata",
                    "|-boost|p1a: Rattata|atk|0",
                    "|move|p2a: Chansey|Tackle|p1a: Rattata",
                    "|-damage|p1a: Rattata|80/100",
                ],
            },
            side_one_choice="swordsdance",
            side_two_choice="tackle",
        )

        self.assertIsNone(legal)
        self.assertEqual(self.loaded_states, [])

    def test_uses_the_first_opponent_hit_affected_by_a_stat_event(self):
        legal = branch_event_legal_rolls(
            {
                "events": [
                    "|move|p1a: Rattata|Aurora Beam|p2a: Chansey",
                    "|-damage|p2a: Chansey|49/100",
                    "|-unboost|p2a: Chansey|atk|1",
                    "|move|p2a: Chansey|Tackle|p1a: Rattata",
                    "|-damage|p1a: Rattata|67/100",
                ],
                "legal_roll_state": "after-drop",
            },
            side_one_choice="aurorabeam",
            side_two_choice="tackle",
        )

        self.assertEqual(legal.target_side, "p1")
        self.assertEqual(legal.damages, {21, 22, 23, 24, 25})
        self.assertEqual(self.loaded_states, ["after-drop"])

    def test_rejects_completed_post_state_without_a_prefix_snapshot(self):
        with self.assertRaises(differential.BranchLegalRollError):
            branch_event_legal_rolls(
                {
                    "events": [
                        "|switch|p2a: Lanturn|Lanturn, L82, M|143/339 tox",
                        "|move|p1a: Sableye|knockoff|p2a: Lanturn",
                        "|-damage|p2a: Lanturn|122/339 tox",
                    ],
                    "post_state": "must-not-load",
                },
                side_one_choice="knockoff",
                side_two_choice="lanturn",
            )

        self.assertEqual(self.loaded_states, [])

    def test_branch_local_support_does_not_reprice_an_earlier_hit(self):
        support = differential.BranchLegalRollSupport(
            target_side="p1",
            event_index=7,
            critical=False,
            damages={21, 22, 23, 24, 25},
        )
        selected = differential.DamageComponent("", -21, 7)
        earlier = differential.DamageComponent("", -41, 6)

        self.assertEqual(
            differential.branch_component_legal_rolls(
                support,
                target_side="p1",
                component=selected,
                pre_legal={41, 42},
            ),
            {21, 22, 23, 24, 25},
        )
        self.assertEqual(
            differential.branch_component_legal_rolls(
                support,
                target_side="p2",
                component=earlier,
                pre_legal={41, 42},
            ),
            {41, 42},
        )

    def test_critical_direct_hit_cannot_borrow_the_noncritical_range(self):
        def calculate_damage(_state, _side_one, _side_two, _first):
            # The Python engine exposes [normal, critical] bases per actor.
            return [25, 100], [30, 120]

        differential.poke_engine.calculate_damage = calculate_damage
        support = branch_event_legal_rolls(
            {
                "events": [
                    "|move|p1a: A|Fire Blast|p2a: B",
                    "|-boost|p1a: A|spa|1",
                    "|-crit|p2a: B",
                    "|-damage|p2a: B|100/200",
                ],
                "legal_roll_state": "post-boost",
            },
            side_one_choice="fireblast",
            side_two_choice="splash",
        )

        self.assertTrue(support.critical)
        self.assertEqual(support.damages, set(range(85, 101)))
        self.assertNotIn(25, support.damages)

    def test_noncritical_direct_hit_cannot_borrow_the_critical_range(self):
        def calculate_damage(_state, _side_one, _side_two, _first):
            return [25, 100], [30, 120]

        differential.poke_engine.calculate_damage = calculate_damage
        support = branch_event_legal_rolls(
            {
                "events": [
                    "|move|p1a: A|Fire Blast|p2a: B",
                    "|-boost|p1a: A|spa|1",
                    "|-damage|p2a: B|100/200",
                ],
                "legal_roll_state": "post-boost",
            },
            side_one_choice="fireblast",
            side_two_choice="splash",
        )

        self.assertFalse(support.critical)
        self.assertEqual(support.damages, {21, 22, 23, 24, 25})
        self.assertNotIn(85, support.damages)

    def test_first_hit_critical_does_not_mark_second_hit_critical(self):
        events = [
            "|move|p1a: A|Double Kick|p2a: B",
            "|-crit|p2a: B",
            "|-damage|p2a: B|150/200",
            "|-damage|p2a: B|125/200",
        ]

        self.assertTrue(
            differential._direct_hit_is_critical(
                events, direct_damage_index=2, target_side="p2"
            )
        )
        self.assertFalse(
            differential._direct_hit_is_critical(
                events, direct_damage_index=3, target_side="p2"
            )
        )

    def test_second_hit_critical_is_local_to_second_hit(self):
        events = [
            "|move|p1a: A|Double Kick|p2a: B",
            "|-damage|p2a: B|175/200",
            "|-crit|p2a: B",
            "|-damage|p2a: B|125/200",
        ]

        self.assertFalse(
            differential._direct_hit_is_critical(
                events, direct_damage_index=1, target_side="p2"
            )
        )
        self.assertTrue(
            differential._direct_hit_is_critical(
                events, direct_damage_index=3, target_side="p2"
            )
        )

    def test_selected_direct_support_cannot_legalize_other_components(self):
        support = differential.BranchLegalRollSupport(
            target_side="p1",
            event_index=7,
            critical=False,
            damages={21, 22, 23, 24, 25},
        )
        pre_legal = {41, 42}
        selected = differential.DamageComponent("", -21, 7)
        same_side_direct = differential.DamageComponent("", -21, 9)
        confusion = differential.DamageComponent("confusion", -21, 8)
        recoil = differential.DamageComponent("recoil", -21, 8)
        drain = differential.DamageComponent("drain", 21, 8)

        self.assertEqual(
            branch_component_legal_rolls(
                support, target_side="p1", component=selected, pre_legal=pre_legal
            ),
            support.damages,
        )
        for component in (same_side_direct, confusion, recoil, drain):
            with self.subTest(component=component):
                self.assertEqual(
                    branch_component_legal_rolls(
                        support,
                        target_side="p1",
                        component=component,
                        pre_legal=pre_legal,
                    ),
                    pre_legal,
                )

    def test_observed_critical_cannot_match_a_noncritical_branch(self):
        support = differential.BranchLegalRollSupport("p1", 7, False, {21, 22, 23, 24, 25})
        self.assertFalse(
            roll_component_events_agree(
                [differential.DamageComponent("", -21, 3, True)],
                [differential.DamageComponent("", -21, 7, False)],
                support=support,
                target_side="p1",
                pre_legal={21, 22, 23, 24, 25},
            )
        )

    def test_observed_noncritical_cannot_match_a_critical_branch(self):
        support = differential.BranchLegalRollSupport("p1", 7, True, {85, 86, 87})
        self.assertFalse(
            roll_component_events_agree(
                [differential.DamageComponent("", -85, 3, False)],
                [differential.DamageComponent("", -85, 7, True)],
                support=support,
                target_side="p1",
                pre_legal={85, 86, 87},
            )
        )




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
        # The resulting HP is deterministic given the inputs: floor((132 + 125) / 2) = 128 for
        # both. The DELTA is not -- it moves with whatever damage landed earlier in the turn,
        # which is why `movepainsplit` is roll-scaled (see the classification test below).
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

    def test_pain_split_is_ROLL_SCALED_because_its_magnitude_inherits_a_roll(self):
        """Pain Split is roll-DEPENDENT, and this test used to assert the opposite.

        It read `test_pain_split_is_compared_EXACTLY_not_roll_scaled`, asserting
        `movepainsplit` landed in the exact bucket with nothing rolled — the behaviour before
        `movepainsplit` was added to `_ROLL_SCALED_SOURCES`. That change was deliberate and its
        reasoning is recorded at `engine_transition_differential.py:310-317`: Pain Split sets both
        mons to `floor((hp_a + hp_b) / 2)`, so its magnitude is a function of the HP left after
        whatever damage landed earlier in the SAME turn. It inherits that hit's roll exactly as a
        capped heal does, and demanding an exact match was a matcher defect that produced the whole
        `I3_roll_inherited` family (reports/c95, reports/c101).

        The test was left behind by that fix and had been failing on main since. Its old name
        asserted the contract backwards, which is why it is renamed rather than edited in place.
        """
        exact, rolled = _split_components([("movepainsplit", -4)])
        self.assertEqual(dict(exact), {}, "movepainsplit must not be in the EXACT bucket")
        self.assertEqual(rolled, [("movepainsplit", -4)])

    def test_pain_splits_roll_window_is_the_currently_shipped_band(self):
        """Roll-scaled is not unbounded — this pins the window that ships TODAY, at its edges.

        The predicate is `abs(eng) * 0.92 - 1` to `abs(eng) * 1.09 + 1`
        (`engine_transition_differential.py:1014-1015`); for an engine magnitude of 4 that is
        [2.68, 5.36], so 5 is the last accepted value and 6 the first rejected.

        Two corrections to an earlier version of this test, both from review:

        1. It used -8 as the only upper case. -8 is 1.75x the engine magnitude, so every subcase
           still passed with the upper coefficient widened from 1.09 to anything under 1.75
           (4k + 1 < 8). The name claimed "and no wider" while measuring nothing of the sort.
           `(-6, False)` is the actual edge and is now included. (The lower edge was already tight:
           -2 against -3 straddles 2.68.)

        2. It called the band "what a floor-divided quantity needs and no more". The repo's own
           record says otherwise and says it is UNRESOLVED:
           `reports/c101_i3_painsplit_tolerance_derivation.json` -- which is headed
           `RETRACTED IN ITS CENTRAL CLAIM`, but the retraction targets its refutation claim, NOT
           the field cited here, and `what_survives.the_floor_argument` keeps the floor reasoning --
           derives
           `|delta| <= ceil(roll_gap / 2)` -- an ABSOLUTE bound, since `d/dx floor((a+x)/2)` is 1/2
           -- and its `still_to_derive` field records that the implementation still needs that
           derivation; #1054's own message calls the proportional band "the wrong SHAPE for this
           class", noting that with no preceding damage Pain Split must match EXACTLY while the
           band would accept +/-(9%+1). So this test pins the shipped window, not a justified one.
        """
        # At engine magnitude 4 the COEFFICIENT and the +/-1 CONSTANT are not separately
        # identified: the upper edge is satisfied by any c in [1.0, 1.25). A large magnitude
        # separates them, because there the constant is negligible and the ratio dominates --
        # engine 100 gives [91.0, 110.0], so 91/110 are the last accepted and 90/111 the first
        # rejected on each side. Both scales are asserted for that reason, not for coverage.
        for engine, showdown, expected in (
            (-4, -3, True),
            (-4, -4, True),
            (-4, -5, True),
            (-4, -2, False),
            (-4, -6, False),
            (-4, -8, False),
            (-100, -91, True),
            (-100, -110, True),
            (-100, -90, False),
            (-100, -111, False),
        ):
            with self.subTest(engine=engine, showdown=showdown):
                self.assertIs(
                    roll_components_agree(
                        [("movepainsplit", showdown)], [("movepainsplit", engine)], None
                    ),
                    expected,
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


if __name__ == "__main__":  # pragma: no cover
    # At the END. It sat at line 592, stranding MajorityBranchOverride, PainSplitSetHp, PairBySource
    # from direct execution -- found by the repo-wide structural guard in
    # tests/test_public_invariant.py.
    unittest.main()


class WrongFanControlMap(unittest.TestCase):
    """Pins the remap behind ``scripts/c134_wrong_fan_control.py``.

    That script answers the question the sweep cannot: enumeration closes four rows
    while inflating the branch count 8.5x-72.5x, and ``evaluate_boundary_strict``
    accepts on the FIRST matching branch, so "nothing opened" is consistent with a real
    fix AND with a lottery. The control gives the matcher a fan of comparable
    cardinality whose values are NOT legal rolls, and requires the rows to stay
    divergent.

    The whole control rests on the remap being an honest wrong fan, so the remap is
    pinned here rather than trusted. Two earlier versions were not honest and the
    measurement said so: a constant down-shift by the fan width dropped every branch on
    a low-HP defender, and clamping that shift left the "wrong" fan OVERLAPPING the
    legal one, i.e. still containing correct rolls.
    """

    @staticmethod
    def _map(legal):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "c134_wrong_fan_control_under_test",
            REPO_ROOT / "scripts" / "c134_wrong_fan_control.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.wrong_fan_map(legal)

    def test_the_wrong_fan_is_disjoint_same_size_and_nearby(self) -> None:
        fans = {
            "contiguous": list(range(103, 123)),
            "sparse crit+non-crit": [187, 189, 191, 193, 195, 198, 200, 202, 204, 206,
                                     209, 211, 213, 215, 217, 220],
            "low hp, span exceeds minimum": [5, 9, 14, 20, 27, 35, 44],
            "singleton": [112],
        }
        for label, legal in fans.items():
            with self.subTest(fan=label):
                mapping = self._map(legal)
                wrong = sorted(mapping.values())
                self.assertEqual(
                    len(wrong), len(legal), "the wrong fan must have the SAME cardinality"
                )
                self.assertEqual(len(set(wrong)), len(wrong), "the remap must be injective")
                self.assertFalse(
                    set(wrong) & set(legal),
                    f"the wrong fan still contains legal rolls: {sorted(set(wrong) & set(legal))}",
                )
                self.assertTrue(all(value > 0 for value in wrong), "damage must stay positive")
                # Nearby, so no branch gains or loses a faint and the control changes
                # exactly one property: whether the values are legal rolls.
                span = max(legal) - min(legal) + 1
                for source, target in mapping.items():
                    self.assertLessEqual(
                        abs(target - source), span + len(legal),
                        f"{source} -> {target} left the fan's magnitude range",
                    )
