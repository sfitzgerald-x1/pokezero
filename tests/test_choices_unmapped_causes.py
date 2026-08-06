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

    def test_no_legal_action_at_all_is_not_a_switch_only_decision(self) -> None:
        """The blocking finding from review, pinned.

        An earlier version folded this into ``all_unmapped_switch_only`` while its own comment
        admitted it was "a different bug entirely". Five distinct inputs reached that token
        without being switch-only decisions: an empty candidate list, all-illegal candidates,
        an out-of-range mask, an all-False mask, and -- worst -- a ``str`` metadata field,
        which is a ``Sequence`` and so walked past the dedicated plumbing guard.

        The counter's entire value is ownership routing. ``switch_only`` sends an operator to
        read the policy; the truth in those cases is a mask or observation-builder bug.
        """
        self.assertEqual(
            _classify_unmapped(
                aggregated={"tackle": 1.0},
                mapped_any=False,
                any_legal_move=False,
                any_legal_switch=False,
            ),
            "no_legal_action_offered",
        )

    def test_switch_only_requires_a_legal_switch_to_actually_exist(self) -> None:
        """The two differ by `any_legal_switch` alone, so they must not collapse either."""
        common: dict[str, Any] = {
            "aggregated": {"tackle": 1.0},
            "mapped_any": False,
            "any_legal_move": False,
        }
        self.assertNotEqual(
            _classify_unmapped(any_legal_switch=True, **common),
            _classify_unmapped(any_legal_switch=False, **common),
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
                    for any_legal_switch in (False, True):
                        reachable.add(
                            _classify_unmapped(
                                aggregated=aggregated,
                                mapped_any=mapped_any,
                                any_legal_move=any_legal_move,
                                any_legal_switch=any_legal_switch,
                            )
                        )
        dead = set(_CHOICES_UNMAPPED_CAUSES) - reachable - {"no_action_candidates"}
        self.assertEqual(dead, set(), f"vocabulary has unreachable token(s) {dead}")


if __name__ == "__main__":
    unittest.main()


class MapChoicesWiringTests(unittest.TestCase):
    """The classifier tests above are NOT enough, and review proved it.

    Every test in the first version of this file called ``_classify_unmapped`` with
    hand-passed booleans. Four mutations therefore survived all 293 tests touching
    ``engine_search``:

      * swapping ``any_legal_move`` and ``any_legal_switch`` at the only production call site
        -- fully inverting the two tokens the PR says have different owners;
      * typoing the ``no_action_candidates`` literal, shipping an unregistered key;
      * deleting either increment entirely, so the counter is always empty.

    A pure-function test cannot see any of those. These drive the real ``_map_choices``.
    """

    @staticmethod
    def _policy(**stats_kwargs: Any) -> Any:
        from pokezero.engine_search import EngineMctsPolicy

        policy = EngineMctsPolicy.__new__(EngineMctsPolicy)
        policy.stats = _StatsStub()
        return policy

    @staticmethod
    def _context(candidates: Any, mask: Sequence[bool]) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            observation=SimpleNamespace(
                metadata={"action_candidates": candidates},
                legal_action_mask=tuple(mask),
            )
        )

    def _run(self, candidates: Any, mask: Sequence[bool], aggregated: Mapping[str, float]) -> str:
        from pokezero.engine_search import EngineMctsPolicy

        policy = self._policy()
        result = EngineMctsPolicy._map_choices(
            policy, self._context(candidates, mask), aggregated
        )
        self.assertIsNone(result, "fixture must reach the None path")
        causes = policy.stats.choices_unmapped_causes
        self.assertEqual(
            sum(causes.values()), 1, f"exactly one cause must be recorded, got {causes}"
        )
        return next(iter(causes))

    def test_a_string_metadata_field_is_plumbing_not_policy(self) -> None:
        """`str` IS a Sequence, so this walked past the plumbing guard into the policy bucket.

        This is the specific input review used to show the misrouting: an operator reading
        `all_unmapped_switch_only` goes to read the policy, when the real fault is that the
        observation handed us a string.
        """
        self.assertEqual(
            self._run("not-a-list", [True] * 9, {"tackle": 1.0}),
            "no_action_candidates",
        )

    def test_a_missing_metadata_field_is_plumbing(self) -> None:
        self.assertEqual(self._run(None, [True] * 9, {"tackle": 1.0}), "no_action_candidates")

    def test_an_empty_candidate_list_offers_no_legal_action(self) -> None:
        # NOT switch_only: the request offered nothing at all.
        self.assertEqual(self._run([], [True] * 9, {"tackle": 1.0}), "no_legal_action_offered")

    def test_an_all_false_mask_offers_no_legal_action(self) -> None:
        candidates = [{"legal": True, "action_index": 0, "kind": "move", "move_id": "tackle"}]
        self.assertEqual(
            self._run(candidates, [False] * 9, {"tackle": 1.0}), "no_legal_action_offered"
        )

    def test_an_out_of_range_action_index_offers_no_legal_action(self) -> None:
        candidates = [{"legal": True, "action_index": 99, "kind": "move", "move_id": "tackle"}]
        self.assertEqual(
            self._run(candidates, [True] * 9, {"tackle": 1.0}), "no_legal_action_offered"
        )

    def test_a_legal_switch_with_no_legal_move_is_switch_only(self) -> None:
        candidates = [
            {"legal": True, "action_index": 4, "kind": "switch",
             "pokemon": {"species": "Blissey"}}
        ]
        self.assertEqual(
            self._run(candidates, [True] * 9, {"tackle": 1.0}), "all_unmapped_switch_only"
        )

    def test_a_different_legal_move_is_a_legality_mismatch(self) -> None:
        candidates = [{"legal": True, "action_index": 0, "kind": "move", "move_id": "surf"}]
        self.assertEqual(
            self._run(candidates, [True] * 9, {"tackle": 1.0}),
            "all_unmapped_legality_mismatch",
        )

    def test_the_two_owner_relevant_tokens_are_not_swapped_at_the_call_site(self) -> None:
        """Kills the mutation that exchanges the two bools where they are computed.

        Asserting each token separately above is not sufficient on its own: a swap inverts
        BOTH, so each individual assertion could in principle be satisfied by the other
        fixture. This states the pairing directly.
        """
        move_only = [{"legal": True, "action_index": 0, "kind": "move", "move_id": "surf"}]
        switch_only = [
            {"legal": True, "action_index": 4, "kind": "switch",
             "pokemon": {"species": "Blissey"}}
        ]
        self.assertEqual(
            (
                self._run(move_only, [True] * 9, {"tackle": 1.0}),
                self._run(switch_only, [True] * 9, {"tackle": 1.0}),
            ),
            ("all_unmapped_legality_mismatch", "all_unmapped_switch_only"),
        )

    def test_an_unregistered_cause_degrades_rather_than_shipping_silently(self) -> None:
        from pokezero.engine_search import _registered_cause_or_unclassified

        self.assertEqual(
            _registered_cause_or_unclassified("no_action_candidate"), "unclassified_cause"
        )
        self.assertEqual(
            _registered_cause_or_unclassified("no_action_candidates"), "no_action_candidates"
        )


class _StatsStub:
    """Only the two counters `_map_choices` touches."""

    def __init__(self) -> None:
        self.unmapped_choices: Counter = Counter()
        self.choices_unmapped_causes: Counter = Counter()
