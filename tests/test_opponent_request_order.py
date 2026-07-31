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

    def order_for(self, actives, party=None):
        determinization._own_observations_by_decision_round = (
            lambda ctx: dict(enumerate(actives))
        )
        determinization._public_opponent_active_species = lambda o: o
        return opponent_request_order(SimpleNamespace(), party or PARTY)

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


if __name__ == "__main__":
    unittest.main()
