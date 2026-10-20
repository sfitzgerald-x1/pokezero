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

from pathlib import Path
import json
import re
from types import SimpleNamespace
import unittest

from pokezero.engine_search import (
    OPPONENT_REQUEST_ORDER_STATUS_VALUES,
    opponent_request_order,
    opponent_request_order_resolution,
)
from pokezero.policy import PolicyContext
from pokezero.trajectory import BattleTrajectory, TrajectoryStep

PARTY = ["typhlosion", "smeargle", "absol", "vaporeon", "sharpedo", "deoxysdefense"]
MASK = (True,) * 9
MOVE_ACTIONS = 4


def observation(active_species, *, recent_events=()):
    return SimpleNamespace(
        metadata={
            "opponent_active": {"species": active_species},
            "recent_public_events": list(recent_events),
        },
        legal_action_mask=MASK,
    )


def step(player_id, turn, active_species, action_index, *, recent_events=()):
    return TrajectoryStep(
        player_id=player_id,
        turn_index=turn,
        observation=observation(active_species, recent_events=recent_events),
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
    def _contradictory_switch_context(self, species, action):
        return context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 5),
             step("p1", 1, "Absol", 0), step("p2", 1, "Absol", action)],
            decision_round=2, active=species,
            switch_rounds=[switch_round(0, "Absol"), switch_round(1, species)],
        )

    def test_failure_witness_preserves_both_kinds_of_constraint_conflict(self) -> None:
        for species, action, index in [
            ("Typhlosion", 4, 1), ("Vaporeon", 5, 0), ("x" * 10000, 5, 0),
        ]:
            ctx = self._contradictory_switch_context(species, action)
            plain = opponent_request_order_resolution(ctx, PARTY)
            observed = opponent_request_order_resolution(
                ctx, PARTY, capture_failure_witness=True,
            )
            with self.subTest(species=species):
                self.assertEqual((observed.order, observed.status), (plain.order, plain.status))
                self.assertEqual(observed.status, "rejected_public_order_walk")
                self.assertIsNone(plain.failure_witness)
                witness = observed.failure_witness
                self.assertEqual(witness["site"], "switch_constraint_conflict")
                self.assertEqual(witness["turn_index"], 1)
                self.assertEqual(witness["species"], species.lower()[:64])
                self.assertEqual(witness["requested_index"], index)
                self.assertEqual(witness["action_index"], action)
                self.assertEqual(witness["constraints"], {"typhlosion": 0, "absol": 2})
                self.assertEqual(witness["current_order"], [2, 1, 0, 3, 4, 5])
                self.assertEqual(witness["active_position"], 0)
                self.assertEqual(witness["evidence_source"], "captured_public_action")
                self.assertFalse(witness["sampled_party_mode"])
                self.assertEqual(witness["sampled_party"], PARTY)
                self.assertEqual(json.loads(json.dumps(witness)), witness)
                self.assertLess(len(json.dumps(witness)), 2048)

    def test_all_walk_rejection_sites_capture_without_changing_the_result(self) -> None:
        from pokezero.determinization import _public_opponent_team_index_walk

        opening = context([step("p1", 0, "Typhlosion", 0)],
                          decision_round=1, active="Typhlosion")
        missing_party = ["typhlosion", "smeargle", "ditto", "vaporeon", "sharpedo", "deoxysdefense"]
        switched = self._contradictory_switch_context("Absol", 0)
        sampled_same = context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 5)],
            decision_round=1, active="Typhlosion",
            switch_rounds=[switch_round(0, "Typhlosion")],
        )
        replaced = context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 0)],
            decision_round=1, active="Absol",
        )
        witnessed_replacement = context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 0)],
            decision_round=1, active="Absol",
        )
        witnessed_replacement.observation.metadata["recent_public_events"] = [
            "|switch|p2a: Absol|Absol, L80|250/250",
        ]
        cases = [
            ("nonpositive_team_size", opening, {"team_size": 0}),
            ("invalid_sampled_party", opening, {"sampled_party_species": ["absol"]}),
            ("initial_constraint_conflict", opening,
             {"sampled_party_species": [PARTY[2], PARTY[1], PARTY[0], *PARTY[3:]]}),
            ("sampled_switch_species_missing", switched,
             {"sampled_party_species": missing_party}),
            ("sampled_switch_to_active", sampled_same, {"sampled_party_species": PARTY}),
            ("switch_constraint_conflict", self._contradictory_switch_context("Typhlosion", 4), {}),
            ("sampled_observed_species_missing", replaced,
             {"sampled_party_species": missing_party}),
            ("replacement_constraint_conflict", witnessed_replacement,
             {"party_index_by_species": {"absol": 0}}),
        ]
        for site, ctx, overrides in cases:
            kwargs = {"opponent_slot": "p2", "team_size": 6, **overrides}
            witness = {}
            with self.subTest(site=site):
                self.assertIsNone(_public_opponent_team_index_walk(ctx, **kwargs))
                self.assertIsNone(_public_opponent_team_index_walk(
                    ctx, **kwargs, failure_witness=witness,
                ))
                self.assertEqual(witness["site"], site)

    def test_successful_walk_has_identical_order_and_no_failure_witness(self) -> None:
        ctx = context(
            [step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 5),
             step("p1", 1, "Absol", 0)],
            decision_round=2, active="Absol", switch_rounds=[switch_round(0, "Absol")],
        )
        self.assertEqual(
            opponent_request_order_resolution(ctx, PARTY),
            opponent_request_order_resolution(ctx, PARTY, capture_failure_witness=True),
        )

    def test_unrelated_order_refusal_has_no_failure_witness(self) -> None:
        ctx = context([step("p1", 0, "Typhlosion", 0)], decision_round=1,
                      active="Typhlosion")
        for party in ([], PARTY + [PARTY[0]]):
            result = opponent_request_order_resolution(ctx, party, capture_failure_witness=True)
            self.assertIsNone(result.failure_witness)

    def test_no_switches_is_the_party_order(self) -> None:
        ctx = context([step("p1", 0, "Typhlosion", 0)], decision_round=1,
                      active="Typhlosion")
        self.assertEqual(opponent_request_order(ctx, PARTY), PARTY)
        resolution = opponent_request_order_resolution(ctx, PARTY)
        self.assertEqual(resolution.order, tuple(PARTY))
        self.assertEqual(resolution.status, "resolved")

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

    def test_public_forced_replacement_restores_the_request_permutation(self) -> None:
        """A public replacement after a move still performs Showdown's slot swap.

        Absol first enters through a recorded voluntary switch.  It then faints
        after its next recorded action is a move, so Vaporeon is brought in by a
        public replacement rather than another opponent switch action.  The
        prior walk discarded the active permutation at this boundary: it only
        knew how to bind a species reached through an action-indexed switch.

        The sampled party is unique, so Vaporeon's original index identifies
        its current request position in the already-tracked permutation.  A
        replacement swaps that position into slot zero just like a voluntary
        switch; the result is therefore recoverable from public evidence, not
        an invented opponent order.
        """
        ctx = context(
            [
                step("p1", 0, "Typhlosion", 0),
                step("p2", 0, "Typhlosion", 5),  # switch to Absol (position 2)
                step("p1", 1, "Absol", 0),
                step("p2", 1, "Absol", 0),       # move; Absol then faints
                step(
                    "p1", 2, "Vaporeon", 0,
                    recent_events=("|switch|p2a: Vaporeon|Vaporeon, L80|250/250",),
                ),
            ],
            decision_round=3,
            active="Vaporeon",
            switch_rounds=[switch_round(0, "Absol")],
        )
        self.assertEqual(
            opponent_request_order(ctx, PARTY),
            ["vaporeon", "smeargle", "typhlosion", "absol", "sharpedo", "deoxysdefense"],
        )

    def test_unwitnessed_active_change_still_refuses_a_party_aware_recovery(self) -> None:
        """A sampled party alone is not evidence that a slot swap happened."""
        ctx = context(
            [
                step("p1", 0, "Typhlosion", 0),
                step("p2", 0, "Typhlosion", 5),
                step("p1", 1, "Absol", 0),
                step("p2", 1, "Absol", 0),
                step("p1", 2, "Vaporeon", 0),
            ],
            decision_round=3,
            active="Vaporeon",
            switch_rounds=[switch_round(0, "Absol")],
        )
        resolution = opponent_request_order_resolution(ctx, PARTY)
        self.assertIsNone(resolution.order)
        self.assertEqual(resolution.status, "lost_active_permutation")

    def test_replacement_after_a_recorded_switch_applies_both_slot_swaps(self) -> None:
        """A same-round switch and faint replacement are two distinct mutations."""
        ctx = context(
            [
                step("p1", 0, "Typhlosion", 0),
                step("p2", 0, "Typhlosion", 5),  # voluntary switch to Absol
                step(
                    "p1", 1, "Vaporeon", 0,
                    recent_events=("|switch|p2a: Vaporeon|Vaporeon, L80|250/250",),
                ),
            ],
            decision_round=2,
            active="Vaporeon",
            switch_rounds=[switch_round(0, "Absol")],
        )
        self.assertEqual(
            opponent_request_order(ctx, PARTY),
            ["vaporeon", "smeargle", "typhlosion", "absol", "sharpedo", "deoxysdefense"],
        )

    def test_cosmetic_unown_form_uses_the_same_recovery_identity(self) -> None:
        """Public Unown forms and sampled Gen 3 source ids share one key space."""
        party = ["typhlosion", "smeargle", "absol", "unownc", "sharpedo", "deoxysdefense"]
        ctx = context(
            [
                step("p1", 0, "Typhlosion", 0),
                step("p2", 0, "Typhlosion", 5),
                step("p1", 1, "Absol", 0),
                step("p2", 1, "Absol", 0),
                step(
                    "p1", 2, "Unown-C", 0,
                    recent_events=("|switch|p2a: Unown-C|Unown-C, L80|250/250",),
                ),
            ],
            decision_round=3,
            active="Unown-C",
            switch_rounds=[switch_round(0, "Absol")],
        )
        self.assertEqual(
            opponent_request_order(ctx, party),
            ["unownc", "smeargle", "typhlosion", "absol", "sharpedo", "deoxysdefense"],
        )

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
    def test_rust_parser_accepts_the_exact_source_status_protocol(self) -> None:
        """A new source refusal category must not die at the native boundary."""

        leaf = (
            Path(__file__).resolve().parents[1]
            / "rust"
            / "pokezero-search"
            / "src"
            / "leaf.rs"
        )
        source = leaf.read_text(encoding="utf-8")
        match = re.search(
            r"const VALID_STATUSES: \[&str; \d+\] = \[(.*?)\];",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, "native request-order status parser disappeared")
        native_statuses = frozenset(re.findall(r'"([a-z_]+)"', match.group(1)))
        self.assertEqual(native_statuses, OPPONENT_REQUEST_ORDER_STATUS_VALUES)

    def test_no_history_at_all_returns_none(self) -> None:
        trajectory = BattleTrajectory(battle_id="b", format_id="gen3randombattle", seed=1)
        ctx = PolicyContext(
            player_id="p1", decision_round_index=0, battle_id="b",
            format_id="gen3randombattle", seed=1, observation=None,
            requested_players=("p1",), trajectory=trajectory,
        )
        self.assertIsNone(opponent_request_order(ctx, PARTY))
        self.assertEqual(
            opponent_request_order_resolution(ctx, PARTY).status,
            "lost_active_permutation",
        )

    def test_duplicate_species_fails_closed(self) -> None:
        party = ["absol", "absol", "smeargle", "vaporeon", "sharpedo", "typhlosion"]
        ctx = context([step("p1", 0, "Absol", 0)], decision_round=1, active="Absol")
        self.assertIsNone(opponent_request_order(ctx, party))
        self.assertEqual(
            opponent_request_order_resolution(ctx, party).status, "duplicate_party"
        )

    def test_cosmetic_forms_of_one_species_are_ambiguous(self) -> None:
        party = ["unownb", "unownc", "smeargle", "vaporeon", "sharpedo", "typhlosion"]
        ctx = context([step("p1", 0, "Unown-B", 0)], decision_round=1, active="Unown-B")
        self.assertIsNone(opponent_request_order(ctx, party))
        self.assertEqual(
            opponent_request_order_resolution(ctx, party).status, "duplicate_party"
        )

    def test_empty_party_fails_closed(self) -> None:
        ctx = context([step("p1", 0, "Typhlosion", 0)], decision_round=1,
                      active="Typhlosion")
        self.assertIsNone(opponent_request_order(ctx, []))
        self.assertEqual(
            opponent_request_order_resolution(ctx, []).status, "empty_party"
        )

    def test_walk_refusal_statuses_are_closed_and_distinct(self) -> None:
        """The native audit must not collapse distinct public-order failures."""
        import pokezero.determinization as determinization

        ctx = context([step("p1", 0, "Typhlosion", 0)], decision_round=1,
                      active="Typhlosion")
        original = determinization._public_opponent_team_index_walk
        cases = {
            "rejected_public_order_walk": None,
            "lost_active_permutation": (None, list(range(len(PARTY))), None),
            "non_permutation_result": (None, [0, 0, 2, 3, 4, 5], 0),
        }
        try:
            for expected, walk_result in cases.items():
                determinization._public_opponent_team_index_walk = (
                    lambda *_args, result=walk_result, **_kwargs: result
                )
                with self.subTest(expected=expected):
                    resolution = opponent_request_order_resolution(ctx, PARTY)
                    self.assertIsNone(resolution.order)
                    self.assertEqual(resolution.status, expected)

            def raises(*_args, **_kwargs):
                raise RuntimeError("fixture walk failure")

            determinization._public_opponent_team_index_walk = raises
            resolution = opponent_request_order_resolution(ctx, PARTY)
            self.assertIsNone(resolution.order)
            self.assertEqual(resolution.status, "public_order_walk_error")
        finally:
            determinization._public_opponent_team_index_walk = original

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
