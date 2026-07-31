"""Gate on the opponent's request order handed to the crate.

Showdown keeps a player's active at request slot 0 and swaps the incoming mon
into slot 0 on each switch-in, so the request order is the party order with one
slot-0 swap accumulated per switch-in from battle start. That order is the
label space of the model's opponent action head, and the crate cannot derive it
(it never receives pre-root protocol lines) -- four in-crate approximations
were each wrong beyond a single switch.

So it is computed here and passed through ctx. A WRONG order silently permutes
opponent switch priors, which is worse than the crate's documented fallback, so
every unresolvable case must return None rather than a guess.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import pokezero.determinization as determinization
from pokezero.engine_search import opponent_request_order

PARTY = ["typhlosion", "smeargle", "absol", "vaporeon", "sharpedo", "deoxysdefense"]


def expected_order(history):
    order = list(PARTY)
    for species in history:
        index = order.index(species)
        if index:
            order[0], order[index] = order[index], order[0]
    return order


class OpponentRequestOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._obs = determinization._own_observations_by_decision_round
        self._act = determinization._public_opponent_active_species

    def tearDown(self) -> None:
        determinization._own_observations_by_decision_round = self._obs
        determinization._public_opponent_active_species = self._act

    def order_for(self, actives, party=None, opponent_turns=None):
        """`opponent_turns` = rounds at which the OPPONENT acted.

        Defaults to exactly the observed rounds, i.e. the benign case. Passing a
        round we did not observe is the real-battle case (a forced replacement
        after a faint) and must fail closed.
        """
        determinization._own_observations_by_decision_round = (
            lambda ctx: dict(enumerate(actives))
        )
        determinization._public_opponent_active_species = lambda o: o
        turns = range(len(actives)) if opponent_turns is None else opponent_turns
        context = SimpleNamespace(
            player_id="p1",
            decision_round_index=10_000,
            trajectory=SimpleNamespace(
                steps=[
                    SimpleNamespace(player_id="p2", turn_index=t, action_index=0)
                    for t in turns
                ]
            ),
        )
        return opponent_request_order(context, party or PARTY)

    def test_no_switches_leaves_the_party_order(self) -> None:
        self.assertEqual(self.order_for(["typhlosion"]), expected_order([]))

    def test_one_switch_swaps_into_slot_zero(self) -> None:
        self.assertEqual(self.order_for(["typhlosion", "absol"]), expected_order(["absol"]))

    def test_three_switches_accumulate(self) -> None:
        # THE case four in-crate attempts got wrong: a single slot-0 swap is
        # right at one switch and transposed beyond, so the swaps must compose.
        history = ["absol", "deoxysdefense", "typhlosion"]
        self.assertEqual(
            self.order_for(["typhlosion", *history]), expected_order(history)
        )
        # And it is genuinely different from the one-swap approximation.
        approximation = list(PARTY)
        self.assertNotEqual(expected_order(history), approximation)

    def test_repeated_actives_are_not_counted_as_switches(self) -> None:
        # The active is observed every round; only CHANGES are switch-ins.
        self.assertEqual(
            self.order_for(["typhlosion", "typhlosion", "absol", "absol"]),
            expected_order(["absol"]),
        )

    def test_unknown_species_fails_closed(self) -> None:
        self.assertIsNone(self.order_for(["typhlosion", "mewtwo"]))

    def test_no_observations_fails_closed(self) -> None:
        self.assertIsNone(self.order_for([]))

    def test_duplicate_species_fails_closed(self) -> None:
        # Slot-0 swaps are resolved by species name, so a duplicated species
        # makes the permutation ambiguous.
        party = ["absol", "absol", "smeargle", "vaporeon", "sharpedo", "typhlosion"]
        self.assertIsNone(self.order_for(["absol"], party=party))


class FailClosedOnUnobservedRoundsTest(OpponentRequestOrderTest):
    """The failure that made attempt 5 wrong on 21% of real decisions.

    The opponent also acts at rounds where WE were not requested -- most often
    a forced replacement after a faint. Those rounds carry no observation on
    our side, so an active-diff under-counts switch-ins and returns a
    confidently wrong permutation that stays broken for the rest of the battle.
    Measured against live Showdown: 170 of 811 decisions across 12 games, and
    it never detected its own error.
    """

    def test_opponent_action_at_an_unobserved_round_fails_closed(self) -> None:
        # We observed rounds 0..2; the opponent also acted at round 3.
        self.assertIsNone(
            self.order_for(
                ["typhlosion", "typhlosion", "absol"], opponent_turns=[0, 1, 2, 3]
            )
        )

    def test_fully_observed_history_still_resolves(self) -> None:
        # The guard must not reject the benign case outright, or the channel is
        # dead code rather than fail-closed.
        self.assertEqual(
            self.order_for(["typhlosion", "absol"], opponent_turns=[0, 1]),
            expected_order(["absol"]),
        )


if __name__ == "__main__":
    unittest.main()
