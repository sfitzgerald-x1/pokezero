import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pokezero.category_vocab import build_category_vocabulary
from pokezero.online_client import (
    LoginError,
    OnlineBattleAgent,
    request_assertion,
    split_server_message,
    to_id,
)
from pokezero.policy import PolicyDecision, legal_action_indices

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown"


def _fixture_lines(name: str) -> list[str]:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8").splitlines()


class _FirstLegalPolicy:
    policy_id = "first-legal"

    def select_action(self, observation, *, rng) -> PolicyDecision:
        action_index = legal_action_indices(observation.legal_action_mask)[0]
        return PolicyDecision(action_index=action_index, policy_id=self.policy_id, action_probability=1.0)


def _agent(our_name: str = "PokeZeroBot", **kwargs) -> OnlineBattleAgent:
    # dex=None keeps type/stat slots padding; a tiny vocab is fine (unknown tokens go to OOV).
    vocab = build_category_vocabulary(["species:Charizard", "move:flamethrower"], oov_buckets=16)
    return OnlineBattleAgent(
        policy=_FirstLegalPolicy(), vocab=vocab, dex=None, our_name=our_name, **kwargs
    )


class SplitServerMessageTest(unittest.TestCase):
    def test_room_frame(self) -> None:
        room, lines = split_server_message(">battle-gen3randombattle-1\n|init|battle\n|request|{}")
        self.assertEqual(room, "battle-gen3randombattle-1")
        self.assertEqual(lines, ["|init|battle", "|request|{}"])

    def test_global_frame(self) -> None:
        room, lines = split_server_message("|challstr|4|abcdef")
        self.assertEqual(room, "")
        self.assertEqual(lines, ["|challstr|4|abcdef"])

    def test_to_id_strips_non_alphanumeric(self) -> None:
        self.assertEqual(to_id("PokeZero Bot!"), "pokezerobot")


class RequestAssertionTest(unittest.TestCase):
    def _mock_urlopen(self, body: str):
        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *args):
                return False

            def read(self_inner):
                return body.encode("utf-8")

        return patch("urllib.request.urlopen", return_value=_Resp())

    def test_unregistered_assertion(self) -> None:
        with self._mock_urlopen("1,assertiondata"):
            self.assertEqual(request_assertion("4|chal", "Guest", None), "1,assertiondata")

    def test_registered_assertion_from_json(self) -> None:
        with self._mock_urlopen("]" + json.dumps({"assertion": "signed-token", "actionsuccess": True})):
            self.assertEqual(request_assertion("4|chal", "Bot", "pw"), "signed-token")

    def test_refused_assertion_raises(self) -> None:
        with self._mock_urlopen(";name is registered"):
            with self.assertRaises(LoginError):
                request_assertion("4|chal", "Guest", None)


class OnlineBattleAgentTest(unittest.TestCase):
    def test_chooses_a_legal_action_from_live_protocol(self) -> None:
        # The fixture is a real p2 move-request log — the same line format a live room streams.
        # Pinned to the v2.1 spec this battery was written against (the toy vocab has no
        # turn-merged families; the v2.2 default path is covered by
        # TurnMergedNormalizeThreadingTest).
        from pokezero.showdown import V2_1_REPLAY_OBSERVATION_SPEC

        choice = _agent(spec=V2_1_REPLAY_OBSERVATION_SPEC).choose(
            _fixture_lines("p2_seat_replay.txt"), "battle-gen3randombattle-1"
        )
        self.assertIsNotNone(choice)
        self.assertRegex(choice, r"^(move|switch) [1-9]$")

    def test_waits_on_a_wait_request(self) -> None:
        wait = {"wait": True, "side": {"id": "p2", "name": "PokeZeroBot", "pokemon": []}}
        lines = ["|player|p2|PokeZeroBot|1|", "|request|" + json.dumps(wait)]
        self.assertIsNone(_agent().choose(lines, "battle-x"))

    def test_unresolvable_seat_returns_none(self) -> None:
        # No request for our name yet -> nothing to choose.
        self.assertIsNone(_agent("Nobody").choose(_fixture_lines("p2_seat_replay.txt"), "battle-x"))


class TurnMergedNormalizeThreadingTest(unittest.TestCase):
    """A v2.2 (turn-merged) agent must normalize with include_turn_merged=True.

    Regression for the foul-play probe hang: a v2.2-spec bot called
    normalize_for_player WITHOUT include_turn_merged, so the state had no
    turn_merged_tokens and observation_from_player_state raised on the first
    move (outside choose()'s try/except) — the bot died and foul-play won every
    game by forfeit, hanging the probe. choose() must thread the flag by schema.
    """

    def _v2_2_agent(self):
        from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V2_2
        from pokezero.showdown import V2_2_REPLAY_OBSERVATION_SPEC

        self.assertEqual(
            V2_2_REPLAY_OBSERVATION_SPEC.schema_version, OBSERVATION_SCHEMA_VERSION_V2_2
        )
        vocab = build_category_vocabulary(["species:Charizard"], oov_buckets=16)
        return OnlineBattleAgent(
            policy=_FirstLegalPolicy(),
            vocab=vocab,
            dex=None,
            our_name="PokeZeroBot",
            spec=V2_2_REPLAY_OBSERVATION_SPEC,
        )

    def test_v2_2_agent_passes_include_turn_merged_true(self) -> None:
        captured: dict = {}

        def fake_normalize(*args, **kwargs):
            captured.update(kwargs)
            raise ValueError("stop after capture")  # short-circuit; choose() catches this

        agent = self._v2_2_agent()
        with patch("pokezero.online_client.normalize_for_player", side_effect=fake_normalize):
            self.assertIsNone(agent.choose(["|player|p1|PokeZeroBot|1"], "room"))
        self.assertTrue(
            captured.get("include_turn_merged"),
            "v2.2 agent must call normalize_for_player(include_turn_merged=True)",
        )

    def test_default_schema_agent_requests_turn_merged(self) -> None:
        # The default spec IS v2.2 since the 2026-07-08 promotion, so an unpinned agent
        # must request turn-merged normalization.
        captured: dict = {}

        def fake_normalize(*args, **kwargs):
            captured.update(kwargs)
            raise ValueError("stop after capture")

        with patch("pokezero.online_client.normalize_for_player", side_effect=fake_normalize):
            self.assertIsNone(_agent().choose(["|player|p1|PokeZeroBot|1"], "room"))
        self.assertTrue(
            captured.get("include_turn_merged"),
            "default-schema (v2.2) agent must request turn-merged tokens",
        )

    def test_explicit_v2_1_agent_does_not_request_turn_merged(self) -> None:
        # The v2.1 path stays covered post-flip: an explicitly v2.1-pinned agent must not
        # force turn-merged normalization.
        from pokezero.showdown import V2_1_REPLAY_OBSERVATION_SPEC

        captured: dict = {}

        def fake_normalize(*args, **kwargs):
            captured.update(kwargs)
            raise ValueError("stop after capture")

        with patch("pokezero.online_client.normalize_for_player", side_effect=fake_normalize):
            self.assertIsNone(
                _agent(spec=V2_1_REPLAY_OBSERVATION_SPEC).choose(
                    ["|player|p1|PokeZeroBot|1"], "room"
                )
            )
        self.assertFalse(
            captured.get("include_turn_merged", False),
            "explicit v2.1 agent must not request turn-merged tokens",
        )


if __name__ == "__main__":
    unittest.main()


class BeliefSetSourceGateTest(unittest.TestCase):
    def test_agent_threads_set_source_and_env_gate_controls_build(self) -> None:
        # Regression (readiness plan WS-2/H6): the online client is the cluster foul-play
        # probes' bot path; without set-source threading, probes evaluate belief-trained nets
        # with candidate features ablated regardless of pod env.
        import os
        from unittest.mock import patch

        from pokezero.online_client import OnlineBattleAgent

        captured: dict[str, object] = {}

        def fake_normalize(replay, *, player_id, player_name, set_source=None, **kwargs):
            captured["set_source"] = set_source
            raise ValueError("stop here")

        agent = OnlineBattleAgent(
            policy=None, vocab=None, dex=None, our_name="PokeZeroBot", set_source="SENTINEL"
        )
        with patch("pokezero.online_client.normalize_for_player", side_effect=fake_normalize):
            self.assertIsNone(agent.choose(["|player|p1|PokeZeroBot|1"], "room"))
        self.assertEqual(captured["set_source"], "SENTINEL")

        with patch.dict(os.environ, {"POKEZERO_BELIEF_SET_SOURCE": "0"}):
            from pokezero.local_showdown import belief_set_source_env_enabled

            self.assertFalse(belief_set_source_env_enabled())
