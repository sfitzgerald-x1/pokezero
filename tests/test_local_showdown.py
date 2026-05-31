import json
import os
from pathlib import Path
import shutil
import unittest

from pokezero.local_showdown import (
    DEFAULT_SHOWDOWN_ROOT,
    LocalShowdownConfig,
    LocalShowdownEnv,
    requested_players_from_requests,
    showdown_seed_from_int,
)
from pokezero.policy import RandomLegalPolicy
from pokezero.rollout import RolloutConfig, RolloutDriver
from pokezero.showdown import normalize_for_player, parse_showdown_replay, showdown_choice_for_action


def request_payload(
    side: str,
    *,
    wait: bool = False,
    force_switch: bool = False,
    trapped: bool = False,
    maybe_trapped: bool = False,
    disabled_moves: bool = False,
) -> dict:
    moves = [
        {"move": "Tackle", "id": "tackle", "disabled": disabled_moves},
        {"move": "Protect", "id": "protect", "disabled": disabled_moves},
    ]
    active = {"moves": moves}
    if trapped:
        active["trapped"] = True
    if maybe_trapped:
        active["maybeTrapped"] = True
    payload = {
        "active": [active],
        "side": {
            "name": f"PokeZero {side}",
            "id": side,
            "pokemon": [
                {
                    "ident": f"{side}: Charizard",
                    "details": "Charizard, L80, M",
                    "condition": "250/250",
                    "active": True,
                },
                {
                    "ident": f"{side}: Snorlax",
                    "details": "Snorlax, L80, M",
                    "condition": "350/350",
                    "active": False,
                },
            ],
        },
    }
    if wait:
        payload = {"wait": True, "side": payload["side"]}
    if force_switch:
        payload = {"forceSwitch": [True], "side": payload["side"]}
    return payload


def replay_lines_with_request(request: dict) -> list[str]:
    side = request["side"]["id"]
    opponent = "p2" if side == "p1" else "p1"
    return [
        "|player|p1|PokeZero p1|",
        "|player|p2|PokeZero p2|",
        f"|switch|{side}a: Charizard|Charizard, L80, M|250/250",
        f"|switch|{opponent}a: Blastoise|Blastoise, L80, M|250/250",
        f"|request|{json.dumps(request, separators=(',', ':'))}",
    ]


def integration_config() -> LocalShowdownConfig | None:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=10.0, idle_timeout_seconds=0.02)


class LocalShowdownRequestTest(unittest.TestCase):
    def test_showdown_seed_from_int_is_four_part_deterministic_seed(self) -> None:
        seed = showdown_seed_from_int(123)

        self.assertEqual(seed, showdown_seed_from_int(123))
        self.assertNotEqual(seed, showdown_seed_from_int(124))
        parts = seed.split(",")
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(0 <= int(part) <= 65535 for part in parts))

    def test_requested_players_from_normal_and_force_switch_requests(self) -> None:
        requests = {
            "p1": request_payload("p1"),
            "p2": request_payload("p2", force_switch=True),
        }

        self.assertEqual(requested_players_from_requests(requests), ("p1", "p2"))

    def test_requested_players_ignores_wait_and_team_preview(self) -> None:
        requests = {
            "p1": request_payload("p1", wait=True),
            "p2": {"teamPreview": True, "side": {"id": "p2"}},
        }

        self.assertEqual(requested_players_from_requests(requests), ())

    def test_choice_translation_allows_moves_and_switches_for_normal_turn(self) -> None:
        replay = parse_showdown_replay(replay_lines_with_request(request_payload("p1")))
        state = normalize_for_player(replay, player_id="p1", configured_showdown_slot="p1")

        self.assertEqual(showdown_choice_for_action(state, 0), "move 1")
        self.assertEqual(showdown_choice_for_action(state, 4), "switch 2")

    def test_choice_translation_blocks_trapped_switches(self) -> None:
        replay = parse_showdown_replay(replay_lines_with_request(request_payload("p1", trapped=True)))
        state = normalize_for_player(replay, player_id="p1", configured_showdown_slot="p1")

        self.assertTrue(state.legal_action_mask[0])
        self.assertFalse(state.legal_action_mask[4])
        with self.assertRaisesRegex(ValueError, "not legal"):
            showdown_choice_for_action(state, 4)

    def test_choice_translation_blocks_maybe_trapped_switches(self) -> None:
        replay = parse_showdown_replay(replay_lines_with_request(request_payload("p1", maybe_trapped=True)))
        state = normalize_for_player(replay, player_id="p1", configured_showdown_slot="p1")

        self.assertFalse(state.legal_action_mask[4])

    def test_choice_translation_rejects_all_disabled_moves(self) -> None:
        replay = parse_showdown_replay(replay_lines_with_request(request_payload("p1", disabled_moves=True)))
        state = normalize_for_player(replay, player_id="p1", configured_showdown_slot="p1")

        self.assertFalse(state.legal_action_mask[0])
        self.assertFalse(state.legal_action_mask[1])
        with self.assertRaisesRegex(ValueError, "not legal"):
            showdown_choice_for_action(state, 0)


@unittest.skipIf(integration_config() is None, "requires node and built Pokemon Showdown checkout")
class LocalShowdownIntegrationTest(unittest.TestCase):
    def test_random_vs_random_rollout_reaches_terminal_or_cap_without_showdown_errors(self) -> None:
        config = integration_config()
        assert config is not None
        with LocalShowdownEnv(config) as env:
            result = RolloutDriver(
                env=env,
                policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                config=RolloutConfig(max_decision_rounds=120),
            ).run(seed=3)
            lines = env.protocol_lines

        self.assertIn(result.terminal.winner, {"p1", "p2", None})
        self.assertGreater(len(result.trajectory.steps), 0)
        for step in result.trajectory.steps:
            self.assertTrue(step.legal_action_mask[step.action_index])
        self.assertFalse(any(line.startswith("|error|") for line in lines))

    def test_same_seed_reproduces_winner_and_first_actions(self) -> None:
        first = self._rollout_summary(seed=11)
        second = self._rollout_summary(seed=11)

        self.assertEqual(first, second)

    def _rollout_summary(self, *, seed: int) -> tuple[str | None, tuple[int, ...]]:
        config = integration_config()
        assert config is not None
        with LocalShowdownEnv(config) as env:
            result = RolloutDriver(
                env=env,
                policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                config=RolloutConfig(max_decision_rounds=120),
            ).run(seed=seed)
        return result.terminal.winner, tuple(step.action_index for step in result.trajectory.steps[:24])


if __name__ == "__main__":
    unittest.main()
