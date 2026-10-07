"""Fail-closed regressions for the refused source-position adapter seams."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.engine_search import opponent_request_order_resolution
from pokezero.mcts_eval.policy_opponent_profile import refusal_diagnostic
from pokezero.mcts_eval.source_root_replay import SourceRootReplayError, source_bound_replay_prefix
from pokezero.public_action_capture import public_action_round_from_protocol_lines
from pokezero.public_decision_corpus import PublicActionIdentifier, PublicActorObservation
from pokezero.public_replay_materializer import (
    PublicReplayError, replay_public_action_rounds, resolve_public_action_identifier,
)
from test_opponent_request_order import PARTY, context, step, switch_round
from test_source_root_replay import _record, _round


FAINT = PublicActionIdentifier(kind="event", event_id="faint-before-action")
LINES = ("|move|p1a: Rayquaza|Extreme Speed|p2a: Absol",
         "|-damage|p2a: Absol|0 fnt", "|faint|p2a: Absol")


def cancelled_record(before=("|turn|5",), delta=LINES):
    prior = _record(turn_index=0, recorded_action_index=0, candidates=[], rounds=())
    prior = replace(prior, observation=replace(prior.observation, acting_player_state={
        "recent_public_events": before, "opponent_active": {"species": "Absol"}}))
    target = _record(turn_index=1, recorded_action_index=0, candidates=[],
        rounds=(_round(0, PublicActionIdentifier(kind="move", move_id="extremespeed"),
                       PublicActionIdentifier(kind="event", event_id="unresolved-public-event")),),
        history=(PublicActorObservation(turn_index=0, observation=prior.observation),))
    return replace(target, observation=replace(target.observation, acting_player_state={
        "recent_public_events": (*before, *delta)}))


class RefusalRepairsTest(unittest.TestCase):
    def test_requested_faint_is_a_cancellation_not_a_selected_move(self):
        actions = public_action_round_from_protocol_lines(LINES, turn_index=0,
            requested_players=("p1", "p2")).actions
        self.assertEqual(actions["p2"], FAINT)
        self.assertEqual(actions["p1"].move_id, "extremespeed")

    def test_passive_unrequested_faint_is_not_a_policy_decision(self):
        actions = public_action_round_from_protocol_lines(LINES[1:], turn_index=0,
            requested_players=("p1",)).actions
        self.assertNotIn("p2", actions)

    def test_move_then_faint_remains_the_observed_move(self):
        lines = ("|move|p2a: Absol|Swords Dance|p2a: Absol", *LINES)
        actions = public_action_round_from_protocol_lines(lines, turn_index=0,
            requested_players=("p1", "p2")).actions
        self.assertEqual(actions["p2"].move_id, "swordsdance")

    def test_forced_replacement_does_not_overwrite_cancelled_action(self):
        actions = public_action_round_from_protocol_lines(
            (*LINES, "|switch|p2a: Glalie|Glalie, L82|265/265"), turn_index=0,
            requested_players=("p1", "p2")).actions
        self.assertEqual(actions["p2"], FAINT)

    def test_source_cancellation_uses_only_retained_public_window(self):
        original = cancelled_record()
        repaired = source_bound_replay_prefix(original, source_records=())
        self.assertEqual(repaired.public_action_rounds[0].actions["p2"], FAINT)
        witness = repaired.repairs[0].public_cancellation_witness
        self.assertFalse(witness["historical_selected_action_recovered"])
        self.assertEqual(witness["public_lines"], list(LINES))
        self.assertEqual(original.public_resolved_action_rounds[0].actions["p2"].event_id,
                         "unresolved-public-event")

    def test_separator_only_overlap_cannot_certify_a_source_round(self):
        with self.assertRaises(SourceRootReplayError):
            source_bound_replay_prefix(cancelled_record(before=("|",)), source_records=())

    def test_next_turn_or_wrong_actor_move_cannot_certify_cancellation(self):
        for delta in (("|turn|6", *LINES), (LINES[0].replace("Extreme Speed", "Surf"), *LINES[1:])):
            with self.subTest(delta=delta), self.assertRaises(SourceRootReplayError):
                source_bound_replay_prefix(cancelled_record(delta=delta), source_records=())

    def test_observed_opponent_action_cannot_be_relabelled_cancellation(self):
        with self.assertRaises(SourceRootReplayError):
            source_bound_replay_prefix(cancelled_record(delta=(
                "|move|p2a: Absol|Swords Dance|p2a: Absol", *LINES)), source_records=())

    def test_cancellation_representative_is_a_legal_move_never_a_switch(self):
        observation = SimpleNamespace(legal_action_mask=(False, True, True, False, True), metadata={
            "action_candidates": [{"action_index": 4, "kind": "switch"},
                                  {"action_index": 2, "kind": "move"},
                                  {"action_index": 1, "kind": "move"}]})
        chosen, proof = resolve_public_action_identifier(observation, FAINT, turn_index=0, player_id="p2")
        self.assertEqual(chosen, 1)
        self.assertIn("verified-public-cancellation", proof.resolution)
        observation.metadata["action_candidates"] = [{"action_index": 4, "kind": "switch"}]
        with self.assertRaises(PublicReplayError):
            resolve_public_action_identifier(observation, FAINT, turn_index=0, player_id="p2")

    def test_live_replay_requires_the_sampled_world_to_reproduce_cancellation(self):
        class Env:
            protocol_lines = ()
            def requested_players(self): return ("p1", "p2")
            def terminal(self): return None
            def observe(self, player):
                return SimpleNamespace(legal_action_mask=(True,), metadata={"action_candidates": [
                    {"action_index": 0, "kind": "move", "move_id": "extremespeed"}]})
            def step(self, actions): self.protocol_lines = self.emitted
        action_round = _round(0, PublicActionIdentifier(kind="move", move_id="extremespeed"), FAINT)
        for emitted, accepted in ((LINES, True), (LINES[:1], False),
                (("|move|p2a: Absol|Protect|p2a: Absol", *LINES), False)):
            env = Env()
            env.emitted = emitted
            with self.subTest(emitted=emitted), patch("pokezero.public_replay_materializer.replay_action_rounds"):
                kwargs = dict(seed=1, format_id="gen3randombattle", public_action_rounds=(action_round,), start_override=None)
                if accepted:
                    self.assertEqual(len(replay_public_action_rounds(env, **kwargs).event_canonicalizations), 1)
                else:
                    with self.assertRaisesRegex(PublicReplayError, "not_reproduced"):
                        replay_public_action_rounds(env, **kwargs)

    def test_sampled_own_party_resolves_newly_dragged_species(self):
        ctx = context([step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 0),
                       step("p1", 1, "Absol", 0)], decision_round=2, active="Absol")
        self.assertIsNone(opponent_request_order_resolution(ctx, PARTY).order)
        resolved = opponent_request_order_resolution(ctx, PARTY, sampled_own_party=True)
        self.assertEqual(resolved.order, ("absol", "smeargle", "typhlosion", *PARTY[3:]))

    def test_sampled_switch_uses_public_species_not_foreign_numeric_slot(self):
        ctx = context([step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 4)],
            decision_round=5, active="Absol", switch_rounds=[switch_round(0, "Absol")])
        resolved = opponent_request_order_resolution(ctx, PARTY, sampled_own_party=True)
        self.assertEqual(resolved.order, ("absol", "smeargle", "typhlosion", *PARTY[3:]))
        impossible = context([step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 4)],
            decision_round=1, active="Typhlosion", switch_rounds=[switch_round(0, "Typhlosion")])
        self.assertIsNone(opponent_request_order_resolution(impossible, PARTY, sampled_own_party=True).order)
        # A real switch followed by a drag back in the same chunk is different:
        # preserve both slot swaps, not an impossible switch-to-self refusal.
        drag_back = context([step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 4),
                             step("p1", 1, "Typhlosion", 0)],
            decision_round=2, active="Typhlosion", switch_rounds=[switch_round(0, "Absol")])
        self.assertEqual(opponent_request_order_resolution(drag_back, PARTY, sampled_own_party=True).order,
                         tuple(PARTY))

    def test_sampled_order_still_refuses_unknown_species_and_duplicate_party(self):
        for active, party in (("Mew", PARTY), ("Typhlosion", [PARTY[0]] * 6)):
            ctx = context([step("p1", 0, "Typhlosion", 0), step("p2", 0, "Typhlosion", 0)],
                          decision_round=1, active=active)
            with self.subTest(active=active):
                self.assertIsNone(opponent_request_order_resolution(ctx, party, sampled_own_party=True).order)

    def test_refusal_diagnostics_survive_wrapper_without_entering_policy_input(self):
        cause = ValueError("legal mismatch")
        witness = {"schema": "policy-opponent-refusal-v1", "diagnostic_only_not_policy_input": True,
                   "node_depth": 1, "public_branch_lines": ["|move|p2a: Scyther|Baton Pass|p2a: Scyther"]}
        cause.policy_opponent_diagnostic = json.dumps(witness)
        wrapper = RuntimeError("search refused")
        wrapper.__cause__ = cause
        self.assertEqual(refusal_diagnostic(wrapper), witness)

    def test_malformed_unlabelled_and_cyclic_diagnostics_do_not_hide_refusal(self):
        error = ValueError("original refusal")
        error.__cause__ = error
        for raw in ("not-json", "[]", "{}", '{"schema":"policy-opponent-refusal-v1"}'):
            error.policy_opponent_diagnostic = raw
            self.assertIsNone(refusal_diagnostic(error))
        self.assertEqual(str(error), "original refusal")
