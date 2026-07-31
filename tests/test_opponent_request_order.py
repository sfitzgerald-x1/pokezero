"""Gate on the opponent request order handed to the crate.

Showdown keeps a player's active at request slot 0 and swaps the incoming mon
into slot 0 on every switch-in, so the request order is the party order with
one slot-0 swap accumulated per switch-in. That order is the label space of the
model's opponent action head, and the crate cannot derive it -- it never
receives pre-root protocol lines.

Five hand-rolled reconstructions were each wrong. The fifth diffed the
opponent's public active across OUR decision rounds and was wrong on 170 of 811
decisions across 12 live games, because the opponent also acts at rounds we are
not requested at (a forced replacement after a faint). So this now REUSES
`determinization._public_opponent_team_index_walk`, which consumes the
opponent's trajectory steps directly and already handles drags and same-chunk
faint replacements.

These tests drive the REAL walk. The previous suite monkeypatched both
determinization helpers away and re-implemented the expected order with the
same algorithm as the code, so it could not have caught the defect -- that is
precisely how the 21%-wrong version shipped with 7 green tests.
"""

from __future__ import annotations

import unittest

from pokezero.engine_search import opponent_request_order
from pokezero.policy import PolicyContext
from pokezero.trajectory import BattleTrajectory, TrajectoryStep

PARTY = ["typhlosion", "smeargle", "absol", "vaporeon", "sharpedo", "deoxysdefense"]
MOVE_ACTIONS = 4


def context_for(steps, *, decision_round, player_id="p1"):
    trajectory = BattleTrajectory(battle_id="b", format_id="gen3randombattle", seed=1)
    for step in steps:
        trajectory.append(step)
    return PolicyContext(
        player_id=player_id,
        decision_round_index=decision_round,
        battle_id="b",
        format_id="gen3randombattle",
        seed=1,
        observation=None,
        requested_players=(player_id,),
        trajectory=trajectory,
    )


class WalkReuseTest(unittest.TestCase):
    """The helper must defer to the determinization walk, and fail closed."""

    def test_returns_none_when_the_walk_cannot_resolve(self) -> None:
        # No observations and no opponent steps: the walk has nothing to
        # anchor on, so there is no order to hand the crate.
        self.assertIsNone(
            opponent_request_order(context_for([], decision_round=0), PARTY)
        )

    def test_duplicate_species_fails_closed(self) -> None:
        # Slot swaps are resolved by species name downstream, so a duplicated
        # species makes the mapping ambiguous regardless of the walk.
        party = ["absol", "absol", "smeargle", "vaporeon", "sharpedo", "typhlosion"]
        self.assertIsNone(
            opponent_request_order(context_for([], decision_round=0), party)
        )

    def test_empty_party_fails_closed(self) -> None:
        self.assertIsNone(opponent_request_order(context_for([], decision_round=0), []))

    def test_a_returned_order_is_always_a_permutation_of_the_party(self) -> None:
        # Whatever the walk concludes, the result must be a rearrangement of
        # the sampled party -- never a dropped or invented species. A caller
        # cannot check this, so the helper must guarantee it.
        order = opponent_request_order(context_for([], decision_round=0), PARTY)
        if order is not None:
            self.assertEqual(sorted(order), sorted(PARTY))

    def test_helper_does_not_reimplement_the_walk(self) -> None:
        # The regression that let a 21%-wrong version ship green: the old suite
        # stubbed determinization out entirely, so the real reconciliation was
        # never exercised. Pin that the helper actually calls it.
        import pokezero.determinization as determinization

        calls = []
        original = determinization._public_opponent_team_index_walk

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        determinization._public_opponent_team_index_walk = spy
        try:
            opponent_request_order(context_for([], decision_round=0), PARTY)
        finally:
            determinization._public_opponent_team_index_walk = original
        self.assertEqual(len(calls), 1, "helper did not consult the determinization walk")
        self.assertEqual(calls[0]["team_size"], len(PARTY))
        self.assertEqual(calls[0]["opponent_slot"], "p2")

    def test_opponent_slot_follows_the_acting_seat(self) -> None:
        import pokezero.determinization as determinization

        seen = []
        original = determinization._public_opponent_team_index_walk

        def spy(*args, **kwargs):
            seen.append(kwargs["opponent_slot"])
            return original(*args, **kwargs)

        determinization._public_opponent_team_index_walk = spy
        try:
            opponent_request_order(
                context_for([], decision_round=0, player_id="p2"), PARTY
            )
        finally:
            determinization._public_opponent_team_index_walk = original
        self.assertEqual(seen, ["p1"])


if __name__ == "__main__":
    unittest.main()
