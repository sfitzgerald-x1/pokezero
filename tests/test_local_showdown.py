from dataclasses import replace
import io
import json
import os
from pathlib import Path
import queue
import shutil
import unittest

from pokezero.local_showdown import (
    DEFAULT_SHOWDOWN_ROOT,
    LocalShowdownConfig,
    LocalShowdownEnv,
    _drain_stderr,
    _drain_stdout,
    _start_players_payload,
    requested_players_from_requests,
    showdown_seed_from_int,
)
from pokezero.observation import ObservationFeatureMasks
from pokezero.env import BattleStartOverride
from pokezero.policy import RandomLegalPolicy
from pokezero.rollout import RolloutConfig, RolloutDriver
from pokezero.showdown import (
    V2_1_REPLAY_OBSERVATION_SPEC,
    normalize_for_player,
    parse_showdown_replay,
    showdown_choice_for_action,
)
from pokezero.showdown_fixture import DEFAULT_GEN3_CUSTOM_FORMAT, FixturePokemon, pack_team


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
    # Pinned to v2.1: these are env-mechanics batteries (custom-game injection,
    # snapshot/restore, reseed) written pre-flip; under the v2.2 default their off-pool
    # fixture species would pollute the shared (cached) turn-merged vocabulary's OOV
    # ledger, which the turn-merged tests and the doc extractor assert stays clean.
    return LocalShowdownConfig(
        showdown_root=root,
        read_timeout_seconds=10.0,
        observation_spec=V2_1_REPLAY_OBSERVATION_SPEC,
    )


def _without_timestamp_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(line for line in lines if not line.startswith("|t:|"))


def _without_timestamp_or_reseed_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        line
        for line in lines
        if not line.startswith("|t:|") and "Reseeded to" not in line
    )


class LocalShowdownRequestTest(unittest.TestCase):
    def test_showdown_seed_from_int_is_four_part_deterministic_seed(self) -> None:
        seed = showdown_seed_from_int(123)

        self.assertEqual(seed, showdown_seed_from_int(123))
        self.assertNotEqual(seed, showdown_seed_from_int(124))
        parts = seed.split(",")
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(0 <= int(part) <= 65535 for part in parts))

    def test_start_players_payload_injects_only_overridden_packed_teams(self) -> None:
        payload = _start_players_payload(
            BattleStartOverride(player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"})
        )

        self.assertEqual(payload["p1"], {"name": "PokeZero p1", "team": "Charizard||||Tackle|||||||"})
        self.assertEqual(payload["p2"], {"name": "PokeZero p2", "team": "Xatu||||Psychic|||||||"})

    def test_battle_start_override_rejects_unknown_missing_empty_and_non_customgame_teams(self) -> None:
        with self.assertRaisesRegex(ValueError, "p1 or p2"):
            BattleStartOverride(player_teams={"p1": "Charizard||||Tackle|||||||", "p3": "Xatu||||Psychic|||||||"})
        with self.assertRaisesRegex(ValueError, "complete p1 and p2"):
            BattleStartOverride(player_teams={"p2": "Xatu||||Psychic|||||||"})
        with self.assertRaisesRegex(ValueError, "non-empty"):
            BattleStartOverride(player_teams={"p1": "Charizard||||Tackle|||||||", "p2": ""})
        with self.assertRaisesRegex(ValueError, "gen3customgame"):
            BattleStartOverride(
                player_teams={"p1": "Charizard||||Tackle|||||||", "p2": "Xatu||||Psychic|||||||"},
                format_id="gen3randombattle",
            )

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

    def test_choice_translation_reports_force_switch_request_kind_for_illegal_move(self) -> None:
        replay = parse_showdown_replay(replay_lines_with_request(request_payload("p1", force_switch=True)))
        state = normalize_for_player(replay, player_id="p1", configured_showdown_slot="p1")

        self.assertEqual(state.request_kind, "force_switch")
        with self.assertRaisesRegex(
            ValueError,
            r"action_index 0 is not legal for the current request \(request_kind=force_switch\)\.",
        ):
            showdown_choice_for_action(state, 0)

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

    def test_boundary_reader_ignores_choice_ack_and_stale_requests_until_ready(self) -> None:
        env = LocalShowdownEnv(LocalShowdownConfig(read_timeout_seconds=1.0))
        env._latest_requests = {
            "p1": request_payload("p1"),
            "p2": request_payload("p2"),
        }
        events = iter(
            [
                {"type": "choice_ack", "player": "p1", "choice": "move 1"},
                None,
                {"type": "ready", "requested": ["p1", "p2"]},
            ]
        )
        calls = []

        def read_event(*, timeout: float):
            calls.append(timeout)
            return next(events)

        env._read_event = read_event  # type: ignore[method-assign]

        env._read_until_boundary()

        self.assertEqual(len(calls), 3)

    def test_drain_threads_tolerate_closed_streams(self) -> None:
        stdout_stream = io.StringIO("first\n")
        stderr_stream = io.StringIO("warning\n")
        output_queue: queue.Queue[str | None] = queue.Queue()
        stdout_stream.close()
        stderr_stream.close()

        _drain_stdout(stdout_stream, output_queue)
        stderr_lines: list[str] = []
        _drain_stderr(stderr_stream, stderr_lines)

        self.assertIsNone(output_queue.get_nowait())
        self.assertEqual(stderr_lines, [])


@unittest.skipIf(integration_config() is None, "requires node and built Pokemon Showdown checkout")
class LocalShowdownIntegrationTest(unittest.TestCase):
    def test_reset_with_start_override_runs_custom_game_with_injected_teams(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Tackle")),)
                ),
            },
        )

        with LocalShowdownEnv(config) as env:
            env.reset_with_start_override(seed=7, start_override=start_override)
            self.assertEqual(env.requested_players(), ("p1", "p2"))
            env.step({"p1": 0, "p2": 0})
            lines = env.protocol_lines

        self.assertTrue(any(line.startswith("|switch|p1a: Charmander|") for line in lines))
        self.assertTrue(any(line.startswith("|switch|p2a: Squirtle|") for line in lines))
        self.assertIn("|move|p1a: Charmander|Ember|p2a: Squirtle", lines)
        self.assertIn("|move|p2a: Squirtle|Water Gun|p1a: Charmander", lines)
        self.assertFalse(any(line.startswith("|error|") for line in lines))

    def test_reset_with_start_override_rejects_format_mismatch(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team((FixturePokemon(species="Charmander", moves=("Tackle",)),)),
                "p2": pack_team((FixturePokemon(species="Squirtle", moves=("Tackle",)),)),
            },
        )

        with LocalShowdownEnv(config) as env:
            with self.assertRaisesRegex(ValueError, DEFAULT_GEN3_CUSTOM_FORMAT):
                env.reset_with_start_override(
                    seed=7,
                    format_id="gen3randombattle",
                    start_override=start_override,
                )

    def test_snapshot_restore_replays_same_branch_from_request_boundary(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Tackle")),)
                ),
            },
        )

        with LocalShowdownEnv(config) as env:
            env.reset_with_start_override(seed=17, start_override=start_override)
            snapshot = env.snapshot()
            prefix_len = len(snapshot.protocol_lines)
            env.step({"p1": 0, "p2": 1})
            first_branch_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])

            env.step({"p1": 1, "p2": 0})
            self.assertGreater(len(env.protocol_lines), prefix_len + len(first_branch_suffix))

            env.restore(snapshot)
            self.assertEqual(env.requested_players(), ("p1", "p2"))
            env.step({"p1": 0, "p2": 1})
            restored_branch_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])

        self.assertEqual(restored_branch_suffix, first_branch_suffix)

    def test_snapshot_restore_reuses_prepared_world_in_fresh_bridge_shell(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Tackle")),)
                ),
            },
        )

        with LocalShowdownEnv(config) as env:
            env.reset_with_start_override(seed=17, start_override=start_override)
            snapshot = env.snapshot()
            prefix_len = len(snapshot.protocol_lines)
            env.step({"p1": 0, "p2": 1})
            expected_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])

            env.reset(seed=19)
            with self.assertRaisesRegex(ValueError, "format does not match"):
                env.restore(snapshot)

            env.reset_with_start_override(seed=19, start_override=start_override)
            self.assertNotEqual(env._battle_token, snapshot.battle_token)
            env.restore(snapshot)
            self.assertEqual(env.requested_players(), ("p1", "p2"))
            env.step({"p1": 0, "p2": 1})
            restored_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])

        self.assertEqual(restored_suffix, expected_suffix)

    def test_snapshot_restore_rebuilds_investment_trackers(self) -> None:
        config = integration_config()
        assert config is not None
        config = replace(
            config,
            set_belief_source=True,
            feature_masks=ObservationFeatureMasks(tier2_investment=True),
        )
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Tackle")),)
                ),
            },
        )

        with LocalShowdownEnv(config) as env:
            env.reset_with_start_override(seed=19, start_override=start_override)
            env.observe("p1")
            self.assertIn("p1", env._investment_trackers)
            snapshot = env.snapshot()
            env.step({"p1": 0, "p2": 0})
            self.assertIn("p1", env._investment_trackers)

            env.restore(snapshot)
            self.assertEqual(env._investment_trackers, {})
            env.observe("p1")
            self.assertIn("p1", env._investment_trackers)

    def test_snapshot_restore_after_terminal_branch_keeps_stream_usable(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Mewtwo", ability="Pressure", moves=("Psychic", "Recover")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Caterpie", ability="Shield Dust", moves=("Tackle",), level=1),)
                ),
            },
        )

        with LocalShowdownEnv(config) as env:
            env.reset_with_start_override(seed=23, start_override=start_override)
            snapshot = env.snapshot()
            terminal_branch = env.step({"p1": 0, "p2": 0})
            self.assertIsNotNone(terminal_branch.terminal)

            env.restore(snapshot)
            self.assertEqual(env.requested_players(), ("p1", "p2"))
            restored_branch = env.step({"p1": 1, "p2": 0})

        self.assertIsNone(restored_branch.terminal)

    def test_reseed_simulator_rng_replays_same_seed_and_diverges_different_seed(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Tackle")),)
                ),
            },
        )

        def run_branch(reseed: int):
            with LocalShowdownEnv(config) as env:
                env.reset_with_start_override(seed=31, start_override=start_override)
                self.assertEqual(env.requested_players(), ("p1", "p2"))
                prefix_len = len(env.protocol_lines)
                env.reseed_simulator_rng(reseed)
                self.assertEqual(env.requested_players(), ("p1", "p2"))
                result = env.step({"p1": 0, "p2": 1})
                suffix = _without_timestamp_or_reseed_lines(env.protocol_lines[prefix_len:])
            return result, suffix

        first_result, first_suffix = run_branch(777)
        same_seed_result, same_seed_suffix = run_branch(777)
        different_seed_result, different_seed_suffix = run_branch(778)

        self.assertIsNone(first_result.terminal)
        self.assertIsNone(same_seed_result.terminal)
        self.assertIsNone(different_seed_result.terminal)
        self.assertEqual(same_seed_suffix, first_suffix)
        self.assertNotEqual(different_seed_suffix, first_suffix)

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
