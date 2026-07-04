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


def _agent(our_name: str = "PokeZeroBot") -> OnlineBattleAgent:
    # dex=None keeps type/stat slots padding; a tiny vocab is fine (unknown tokens go to OOV).
    vocab = build_category_vocabulary(["species:Charizard", "move:flamethrower"], oov_buckets=16)
    return OnlineBattleAgent(policy=_FirstLegalPolicy(), vocab=vocab, dex=None, our_name=our_name)


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
        choice = _agent().choose(_fixture_lines("p2_seat_replay.txt"), "battle-gen3randombattle-1")
        self.assertIsNotNone(choice)
        self.assertRegex(choice, r"^(move|switch) [1-9]$")

    def test_waits_on_a_wait_request(self) -> None:
        wait = {"wait": True, "side": {"id": "p2", "name": "PokeZeroBot", "pokemon": []}}
        lines = ["|player|p2|PokeZeroBot|1|", "|request|" + json.dumps(wait)]
        self.assertIsNone(_agent().choose(lines, "battle-x"))

    def test_unresolvable_seat_returns_none(self) -> None:
        # No request for our name yet -> nothing to choose.
        self.assertIsNone(_agent("Nobody").choose(_fixture_lines("p2_seat_replay.txt"), "battle-x"))


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
