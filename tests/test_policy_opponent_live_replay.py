"""Live input qualification does not qualify search or hide missing roots."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.mcts_eval import policy_opponent_live_replay as live
from pokezero.public_replay_materializer import PublicReplayError
from tests.test_policy_opponent_source_prefixes import fixture


class LiveReplayPreflightTests(unittest.TestCase):
    def test_denominator_preserves_source_and_live_refusals_without_redraw(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path, digest, roster = fixture(base)
            calls = []
            def replay(env, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(terminal=None, requested_players=("p1", "p2"))
            def validate(env, record, materialized):
                if record.seed == 2026100100 and record.acting_player == "p1" and record.turn_index == 9:
                    raise PublicReplayError("live_public_observation_differs_from_source")
                return {"simultaneous_continuation_eligible": True}
            with patch.object(live, "replay_public_action_rounds", replay), patch.object(live, "validate_live_root", validate):
                result = live.qualify_live_replay(path, expected_roster_sha256=digest, source_root=base, env=object())
            self.assertEqual(len(calls), 31)
            self.assertEqual([row["decision_id"] for row in result["roots"]],
                             [row["decision_id"] for row in roster["profile_roots"]])
            self.assertEqual((result["root_denominator"], result["live_replay_valid"], result["refused"]), (32, 30, 2))
            self.assertEqual({row["phase"] for row in result["roots"] if row["state"] == "REFUSED"},
                             {"source_prefix", "live_replay"})
            for field in ("live_replay_qualified", "native_worlds_qualified", "profile_qualified", "search_invoked"):
                self.assertFalse(result[field])
            self.assertEqual(result["replacement_roots"], [])

    def test_live_refusal_cannot_hide_source_integrity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path, digest, roster = fixture(base)
            target = base / roster["profile_roots"][0]["source_relative_path"]
            def replay(env, **kwargs):
                if kwargs["seed"] == 2026100100:
                    target.write_bytes(target.read_bytes() + b"\n")
                raise PublicReplayError("sampled_world_request_shape_mismatch")
            with patch.object(live, "replay_public_action_rounds", replay):
                with self.assertRaisesRegex(ValueError, "source bytes drift"):
                    live.qualify_live_replay(path, expected_roster_sha256=digest, source_root=base, env=object())

    def test_history_belief_mask_and_request_boundary_are_all_required(self):
        public = SimpleNamespace()
        observation = SimpleNamespace(metadata={"belief_view": {"x": 1}}, legal_action_mask=(True, False))
        record = SimpleNamespace(acting_player="p1", observation=public, public_belief_view={"x": 1},
                                 current_legal_action_mask=(True, False),
                                 history=(SimpleNamespace(turn_index=0, observation=public),))
        replay = SimpleNamespace(terminal=None, requested_players=("p1",),
                                  replay_observations={0: {"p1": observation}}, event_canonicalizations=())
        env = SimpleNamespace(observe=lambda player: observation, _state_for_player=lambda player: object())
        with patch.object(live.PublicObservation, "from_observation", return_value=public), \
                patch.object(live, "showdown_choice_for_action", return_value="move 1"):
            receipt = live.validate_live_root(env, record, replay)
            self.assertFalse(receipt["simultaneous_continuation_eligible"])
            self.assertEqual(receipt["actor_history_observations_verified"], 1)
            self.assertEqual(receipt["legal_choices"], [{"action_index": 0, "showdown_choice": "move 1"}])
            for attribute, value, reason in (("terminal", object(), "request_boundary"),
                                            ("replay_observations", {}, "actor_history")):
                old = getattr(replay, attribute)
                setattr(replay, attribute, value)
                with self.assertRaisesRegex(PublicReplayError, reason):
                    live.validate_live_root(env, record, replay)
                setattr(replay, attribute, old)
            observation.metadata = {"belief_view": {"x": 2}}
            with self.assertRaisesRegex(PublicReplayError, "public_belief"):
                live.validate_live_root(env, record, replay)
            observation.metadata = {"belief_view": {"x": 1}}
            observation.legal_action_mask = (False, True)
            with self.assertRaisesRegex(PublicReplayError, "legal_action_mask"):
                live.validate_live_root(env, record, replay)

    def test_public_observation_mismatch_refuses_before_legal_serialization(self):
        record = SimpleNamespace(acting_player="p1", observation=object())
        env = SimpleNamespace(observe=lambda player: object())
        replay = SimpleNamespace(terminal=None, requested_players=("p1", "p2"))
        with patch.object(live.PublicObservation, "from_observation", return_value=object()), \
                patch.object(live, "showdown_choice_for_action") as serialize:
            with self.assertRaisesRegex(PublicReplayError, "public_observation"):
                live.validate_live_root(env, record, replay)
            serialize.assert_not_called()
