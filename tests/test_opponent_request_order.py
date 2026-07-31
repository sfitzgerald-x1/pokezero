"""Gate on the opponent request order handed to the crate.

Showdown keeps a player's active at request slot 0 and swaps the incoming mon
into slot 0 on every switch-in, so the request order is the party order with
one slot-0 swap accumulated per switch-in. That order is the label space of the
model's opponent action head, and the crate cannot derive it -- it never
receives pre-root protocol lines.

Five hand-rolled reconstructions were each wrong; the sixth reuses
`determinization._public_opponent_team_index_walk`, which already maintained
exactly this permutation while decoding recorded opponent switch actions.

TEST DESIGN, and why it is what it is. Two previous suites for this helper
gated nothing:

* the first monkeypatched both determinization helpers away and re-implemented
  the expected order with the same algorithm as the code under test;
* the second drove only the fail-closed branch, so it never observed a
  non-None order at all -- inverting the permutation, or deleting the walk's
  swap entirely, left it fully green while live output went 84% and 4% wrong.

So these tests build REAL contexts that make the walk succeed, and assert
concrete orders. The fixtures are internally consistent on purpose: a switch to
the mon at request position p is action index 4 + (p - 1), because that is how
Showdown encodes it and the walk decodes it. An inconsistent fixture tests
nothing.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from pokezero.engine_search import opponent_request_order
from pokezero.policy import PolicyContext
from pokezero.trajectory import BattleTrajectory, TrajectoryStep

PARTY = ["typhlosion", "smeargle", "absol", "vaporeon", "sharpedo", "deoxysdefense"]
MASK = (True,) * 9
MOVE_ACTIONS = 4


def observation(active_species):
    return SimpleNamespace(
        metadata={"opponent_active": {"species": active_species}},
        legal_action_mask=MASK,
    )


def step(player_id, turn, active_species, action_index):
    return TrajectoryStep(
        player_id=player_id,
        turn_index=turn,
        observation=observation(active_species),
        legal_action_mask=MASK,
        action_index=action_index,
        metadata={},
    )


def switch_round(turn, species):
    return {"turn_index": turn, "actions": {"p2": {"kind": "switch",
                                                   "switched_species": species}}}


def context(steps, *, decision_round, active, switch_rounds=(), player_id="p1"):
    trajectory = BattleTrajectory(
        battle_id="b", format_id="gen3randombattle", seed=1,
        metadata={"public_resolved_action_rounds": list(switch_rounds)},
    )
    for item in steps:
        trajectory.append(item)
    return PolicyContext(
        player_id=player_id,
        decision_round_index=decision_round,
        battle_id="b",
        format_id="gen3randombattle",
        seed=1,
        observation=observation(active),
        requested_players=(player_id,),
        trajectory=trajectory,
    )


class RequestOrderTest(unittest.TestCase):
    def test_no_switches_is_the_party_order(self) -> None:
        ctx = context([step("p1", 0, "Typhlosion", 0)], decision_round=1,
                      active="Typhlosion")
        self.assertEqual(opponent_request_order(ctx, PARTY), PARTY)

    def test_one_switch_moves_the_incoming_mon_to_slot_zero(self) -> None:
        # Absol is at request position 2, so its switch action index is 5.
        ctx = context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 5),
             step("p1", 1, "Absol", 0)],
            decision_round=2, active="Absol",
            switch_rounds=[switch_round(0, "Absol")],
        )
        self.assertEqual(
            opponent_request_order(ctx, PARTY),
            ["absol", "smeargle", "typhlosion", "vaporeon", "sharpedo", "deoxysdefense"],
        )

    def test_two_switches_compose(self) -> None:
        """THE case that broke five attempts.

        A single slot-0 swap is right at one switch and transposed beyond, so
        the swaps must accumulate. After absol then deoxys, typhlosion must sit
        at position 2 (not 0) and absol at 5 (not 2).
        """
        ctx = context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 5),
             step("p1", 1, "Absol", 0), step("p2", 1, "Absol", 8),
             step("p1", 2, "Deoxys-Defense", 0)],
            decision_round=3, active="Deoxys-Defense",
            switch_rounds=[switch_round(0, "Absol"), switch_round(1, "Deoxys-Defense")],
        )
        order = opponent_request_order(ctx, PARTY)
        self.assertEqual(
            order,
            ["deoxysdefense", "smeargle", "typhlosion", "vaporeon", "sharpedo", "absol"],
        )
        # And it is NOT the one-swap approximation, or the fixture proves
        # nothing about accumulation.
        one_swap = list(PARTY)
        one_swap[0], one_swap[5] = one_swap[5], one_swap[0]
        self.assertNotEqual(order, one_swap)

    def test_swap_is_load_bearing_when_reconciliation_cannot_repair(self) -> None:
        """Kills the mutation the previous suite missed.

        The walk both SWAPS `current_order` on a decoded switch and, at the
        next observed boundary, RECONCILES the permutation against the species
        actually active. On a fully-observed line the reconciliation can repair
        a missing swap, so deleting the swap leaves most fixtures green -- round
        6 found exactly that mutation surviving.

        Here the round after the switch has no observation on our side, so
        `next_active` is None and the reconciliation is skipped. The order then
        depends purely on the swap: with it, absol reaches slot 0; without it,
        the result is the untouched party order.
        """
        ctx = context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 5)],
            decision_round=5, active="Absol",
            switch_rounds=[switch_round(0, "Absol")],
        )
        order = opponent_request_order(ctx, PARTY)
        self.assertEqual(
            order,
            ["absol", "smeargle", "typhlosion", "vaporeon", "sharpedo", "deoxysdefense"],
        )
        self.assertNotEqual(order, PARTY, "the swap did not happen")

    def test_result_is_always_a_permutation_of_the_party(self) -> None:
        ctx = context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 5),
             step("p1", 1, "Absol", 0)],
            decision_round=2, active="Absol",
            switch_rounds=[switch_round(0, "Absol")],
        )
        order = opponent_request_order(ctx, PARTY)
        self.assertIsNotNone(order)
        self.assertEqual(sorted(order), sorted(PARTY))

    def test_orientation_is_not_inverted(self) -> None:
        """An inverted permutation is self-consistent and passes shape checks.

        Round 6 measured the inverted form wrong on 84% of live decisions, so
        pin the direction with a case where the two differ.
        """
        ctx = context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 5),
             step("p1", 1, "Absol", 0), step("p2", 1, "Absol", 8),
             step("p1", 2, "Deoxys-Defense", 0)],
            decision_round=3, active="Deoxys-Defense",
            switch_rounds=[switch_round(0, "Absol"), switch_round(1, "Deoxys-Defense")],
        )
        order = opponent_request_order(ctx, PARTY)
        inverse = [None] * len(PARTY)
        for position, species in enumerate(order):
            inverse[PARTY.index(species)] = PARTY[position]
        self.assertNotEqual(order, inverse, "fixture cannot distinguish the directions")
        self.assertEqual(order[0], "deoxysdefense", "the ACTIVE must hold slot 0")


class FailClosedTest(unittest.TestCase):
    def test_no_history_at_all_returns_none(self) -> None:
        trajectory = BattleTrajectory(battle_id="b", format_id="gen3randombattle", seed=1)
        ctx = PolicyContext(
            player_id="p1", decision_round_index=0, battle_id="b",
            format_id="gen3randombattle", seed=1, observation=None,
            requested_players=("p1",), trajectory=trajectory,
        )
        self.assertIsNone(opponent_request_order(ctx, PARTY))

    def test_duplicate_species_fails_closed(self) -> None:
        party = ["absol", "absol", "smeargle", "vaporeon", "sharpedo", "typhlosion"]
        ctx = context([step("p1", 0, "Absol", 0)], decision_round=1, active="Absol")
        self.assertIsNone(opponent_request_order(ctx, party))

    def test_empty_party_fails_closed(self) -> None:
        ctx = context([step("p1", 0, "Typhlosion", 0)], decision_round=1,
                      active="Typhlosion")
        self.assertIsNone(opponent_request_order(ctx, []))

    def test_helper_defers_to_the_determinization_walk(self) -> None:
        # Pins that this is a reuse, not a seventh reconstruction.
        import pokezero.determinization as determinization

        calls = []
        original = determinization._public_opponent_team_index_walk

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        determinization._public_opponent_team_index_walk = spy
        try:
            opponent_request_order(
                context([step("p1", 0, "Typhlosion", 0)], decision_round=1,
                        active="Typhlosion"),
                PARTY,
            )
        finally:
            determinization._public_opponent_team_index_walk = original
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["opponent_slot"], "p2")
        self.assertEqual(calls[0]["team_size"], len(PARTY))


if __name__ == "__main__":
    unittest.main()
