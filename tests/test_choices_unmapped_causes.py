"""`choices_unmapped` was 29 fallback decisions on era 60 and said nothing about WHY.

GOAL.md requires `choices_unmapped` at zero INDEPENDENTLY of the fallback rate, so it is a
stop-condition term rather than a rounding error -- and it had one opaque literal across
three call sites, all of the form ``_map_choices(...) is None``. Era 60 could report the
class at 29 and not one word about which of five distinct causes produced it.

Four of those causes have different OWNERS, which is the point:

  * ``no_action_candidates``            -- plumbing: the observation lacks the field.
  * ``aggregated_empty``                -- search produced nothing to map.
  * ``all_unmapped_switch_only``        -- policy: engine proposed moves, none legal.
  * ``all_unmapped_legality_mismatch``  -- belief/PP: a DIFFERENT move was legal.
  * ``mapped_but_no_positive_weight``   -- zero-visit search; nothing was unmapped at all.

The last is invisible to the pre-existing ``unmapped_choices`` counter, because no choice
failed to map -- only the ``weight > best_weight`` comparison against the 0.0 seed did.
That is why this test exists rather than a query over ``unmapped_choices``.
"""

from __future__ import annotations

import unittest
from collections import Counter
from typing import Any, Mapping, Optional, Sequence

from pokezero.engine_search import _CHOICES_UNMAPPED_CAUSES, _classify_unmapped


class ClassifierCauseTests(unittest.TestCase):
    """One assertion per cause, so a mutation that collapses two is caught."""

    def test_no_choices_to_map_is_not_a_mapping_failure(self) -> None:
        self.assertEqual(
            _classify_unmapped(
                aggregated={},
                mapped_any=False,
                any_legal_move=True,
                any_legal_switch=True,
            ),
            "aggregated_empty",
        )

    def test_a_mapped_choice_with_no_positive_weight_blames_the_weight(self) -> None:
        # Nothing was unmapped, so `unmapped_choices` stays empty and this cause is
        # invisible without the counter. A zero-visit search reaches exactly here.
        self.assertEqual(
            _classify_unmapped(
                aggregated={"tackle": 0.0},
                mapped_any=True,
                any_legal_move=True,
                any_legal_switch=True,
            ),
            "mapped_but_no_positive_weight",
        )

    def test_no_legal_move_is_a_switch_only_decision(self) -> None:
        # A force switch, or an active mon with no usable move. The engine proposed moves;
        # the request offered none. Owner: the policy, which should propose a switch.
        self.assertEqual(
            _classify_unmapped(
                aggregated={"tackle": 1.0},
                mapped_any=False,
                any_legal_move=False,
                any_legal_switch=True,
            ),
            "all_unmapped_switch_only",
        )

    def test_a_legal_move_that_is_not_the_proposed_one_is_a_legality_mismatch(self) -> None:
        # PP exhaustion, Taunt, Disable, or a choice/Encore lock: the engine's world and the
        # request disagree about WHICH move is legal. Owner: belief / PP derivation.
        self.assertEqual(
            _classify_unmapped(
                aggregated={"tackle": 1.0},
                mapped_any=False,
                any_legal_move=True,
                any_legal_switch=False,
            ),
            "all_unmapped_legality_mismatch",
        )

    def test_switch_only_and_legality_mismatch_differ_only_by_any_legal_move(self) -> None:
        """The two policy-relevant causes must not collapse.

        They are one bool apart, which is exactly the pair a mutation is most likely to
        merge -- and merging them would re-create era 60's situation with extra steps,
        since they have different owners and different fixes.
        """
        common: dict[str, Any] = {
            "aggregated": {"tackle": 1.0},
            "mapped_any": False,
            "any_legal_switch": True,
        }
        self.assertNotEqual(
            _classify_unmapped(any_legal_move=True, **common),
            _classify_unmapped(any_legal_move=False, **common),
        )


class VocabularyTests(unittest.TestCase):
    def test_every_classifier_output_is_registered(self) -> None:
        """A token the vocabulary does not contain cannot be aggregated across an era.

        Same discipline as the renderer's `UNRENDERABLE_FAMILY_ORDER`: an unregistered
        token is how a class silently stops being rankable. This enumerates the real
        input space rather than sampling it -- five booleans-and-a-mapping, 2**3 * 2
        combinations, all of them.
        """
        seen = set()
        for aggregated in ({}, {"tackle": 1.0}):
            for mapped_any in (False, True):
                for any_legal_move in (False, True):
                    for any_legal_switch in (False, True):
                        seen.add(
                            _classify_unmapped(
                                aggregated=aggregated,
                                mapped_any=mapped_any,
                                any_legal_move=any_legal_move,
                                any_legal_switch=any_legal_switch,
                            )
                        )
        unregistered = seen - set(_CHOICES_UNMAPPED_CAUSES)
        self.assertEqual(
            unregistered,
            set(),
            f"classifier emitted unregistered token(s) {unregistered}",
        )

    def test_the_vocabulary_has_no_dead_tokens_beyond_the_plumbing_one(self) -> None:
        """The reverse direction, which the test above cannot see.

        `no_action_candidates` is emitted at the early return in `_map_choices`, not by the
        classifier, so it is the one legitimate member the exhaustive sweep never produces.
        Everything else must be reachable, or the vocabulary is documenting causes that
        cannot happen -- which is how a reader comes to believe a cause was ruled out when
        it was never wired up.
        """
        reachable = set()
        for aggregated in ({}, {"tackle": 1.0}):
            for mapped_any in (False, True):
                for any_legal_move in (False, True):
                    reachable.add(
                        _classify_unmapped(
                            aggregated=aggregated,
                            mapped_any=mapped_any,
                            any_legal_move=any_legal_move,
                            any_legal_switch=True,
                        )
                    )
        dead = set(_CHOICES_UNMAPPED_CAUSES) - reachable - {"no_action_candidates"}
        self.assertEqual(dead, set(), f"vocabulary has unreachable token(s) {dead}")


if __name__ == "__main__":
    unittest.main()
