from collections.abc import Mapping
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import queue
import shutil
from types import SimpleNamespace
import unittest
from unittest import mock
from typing import Any

import pokezero.local_showdown as local_showdown_module
from pokezero.local_showdown import (
    DEFAULT_SHOWDOWN_ROOT,
    LocalShowdownConfig,
    LocalShowdownError,
    LocalShowdownEnv,
    LocalShowdownSnapshot,
    _drain_stderr,
    _drain_stdout,
    _public_materialization_payload,
    _start_players_payload,
    requested_players_from_requests,
    showdown_seed_from_int,
)
from pokezero.observation import ObservationFeatureMasks
from pokezero.env import BattleStartOverride
from pokezero.policy import RandomLegalPolicy
from pokezero.rollout import RolloutConfig, RolloutDriver
from pokezero.search import (
    prepare_direct_materialization_prefix,
    puct_branch_search,
    release_prepared_replay_prefix,
)
from pokezero.showdown import (
    V2_1_REPLAY_OBSERVATION_SPEC,
    normalize_for_player,
    parse_showdown_replay,
    showdown_choice_for_action,
)
from pokezero.showdown_fixture import DEFAULT_GEN3_CUSTOM_FORMAT, FixturePokemon, pack_team
from pokezero.trajectory import BattleTrajectory


def _active_hp_from_snapshot(snapshot: LocalShowdownSnapshot, player: str) -> int:
    battle = snapshot.bridge_snapshot["battle"]
    side_index = 0 if player == "p1" else 1
    hp = battle["sides"][side_index]["pokemon"][0]["hp"]
    assert isinstance(hp, int)
    return hp


def _active_item_state_from_snapshot(snapshot: LocalShowdownSnapshot, player: str) -> tuple[str, str]:
    battle = snapshot.bridge_snapshot["battle"]
    side_index = 0 if player == "p1" else 1
    pokemon = battle["sides"][side_index]["pokemon"][0]
    current_item = pokemon["item"]
    assigned_item = pokemon["set"]["item"]
    assert isinstance(current_item, str)
    assert isinstance(assigned_item, str)
    return current_item, assigned_item


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

    @staticmethod
    def _winner_slot_env(players: Mapping[str, str]) -> LocalShowdownEnv:
        env = object.__new__(LocalShowdownEnv)
        env._parser = SimpleNamespace(players=dict(players))
        env._sync_incremental_state = lambda: None  # type: ignore[method-assign]
        return env

    def test_winner_slot_resolves_distinct_usernames(self) -> None:
        # Reachable-path control: the default self-play username map resolves each
        # |win| to the right seat and never trips the defensive asserts.
        env = self._winner_slot_env({"p1": "PokeZero p1", "p2": "PokeZero p2"})
        self.assertEqual(env._winner_slot("PokeZero p2"), "p2")
        self.assertEqual(env._winner_slot("PokeZero p1"), "p1")

    def test_winner_slot_asserts_on_unmapped_winner_name(self) -> None:
        # An unmapped winner name would otherwise silently return None ->
        # downgrade a real win to a 0/0 draw target. It must fail loudly instead.
        env = self._winner_slot_env({"p1": "PokeZero p1", "p2": "PokeZero p2"})
        with self.assertRaisesRegex(AssertionError, "did not resolve to a player slot"):
            env._winner_slot("Nobody")

    def test_winner_slot_asserts_on_duplicate_usernames(self) -> None:
        # Two seats sharing a username would first-match every win to one slot —
        # the catastrophic silent mis-attribution the distinctness assert guards.
        env = self._winner_slot_env({"p1": "Twin", "p2": "Twin"})
        with self.assertRaisesRegex(AssertionError, "must be distinct"):
            env._winner_slot("Twin")


class BranchObservationTimingTest(unittest.TestCase):
    def test_state_normalization_substages_have_exact_timing_boundaries(self) -> None:
        env = object.__new__(LocalShowdownEnv)
        replay = object()
        state = object()
        belief_engine = object()
        env._parser = mock.Mock()
        env.config = SimpleNamespace(observation_spec=V2_1_REPLAY_OBSERVATION_SPEC)
        env._observation_format_id = "gen3randombattle"
        env._belief_engine = belief_engine
        env._root_puct_branch_observation_incremental_sync_seconds = 0.0
        env._root_puct_branch_observation_incremental_sync_count = 0
        env._root_puct_branch_observation_replay_snapshot_seconds = 0.0
        env._root_puct_branch_observation_replay_snapshot_count = 0
        env._root_puct_branch_observation_player_state_normalization_seconds = 0.0
        env._root_puct_branch_observation_player_state_normalization_count = 0
        env._root_puct_branch_observation_state_annotation_seconds = 0.0
        env._root_puct_branch_observation_state_annotation_count = 0

        def timed_sync() -> None:
            local_showdown_module.time.perf_counter()

        def timed_snapshot() -> object:
            local_showdown_module.time.perf_counter()
            return replay

        def timed_normalize(*args: object, **kwargs: object) -> object:
            local_showdown_module.time.perf_counter()
            return state

        def timed_tracker(_: str) -> None:
            local_showdown_module.time.perf_counter()
            return None

        env._sync_incremental_state = timed_sync  # type: ignore[method-assign]
        env._parser.snapshot.side_effect = timed_snapshot
        env._tier2_tracker_for = timed_tracker  # type: ignore[method-assign]
        env._investment_tracker_for = timed_tracker  # type: ignore[method-assign]

        with (
            mock.patch(
                "pokezero.local_showdown.time.perf_counter", side_effect=range(13)
            ),
            mock.patch("pokezero.local_showdown.normalize_for_player", side_effect=timed_normalize) as normalizer,
        ):
            self.assertIs(env._state_for_player("p1", root_puct_branch_observation=True), state)

        normalizer.assert_called_once_with(
            replay,
            player_id="p1",
            configured_showdown_slot="p1",
            format_id="gen3randombattle",
            belief_engine=belief_engine,
            include_turn_merged=False,
        )
        self.assertEqual(env._root_puct_branch_observation_incremental_sync_seconds, 2.0)
        self.assertEqual(env._root_puct_branch_observation_replay_snapshot_seconds, 2.0)
        self.assertEqual(env._root_puct_branch_observation_player_state_normalization_seconds, 2.0)
        self.assertEqual(env._root_puct_branch_observation_state_annotation_seconds, 3.0)
        self.assertEqual(env._root_puct_branch_observation_incremental_sync_count, 1)
        self.assertEqual(env._root_puct_branch_observation_replay_snapshot_count, 1)
        self.assertEqual(env._root_puct_branch_observation_player_state_normalization_count, 1)
        self.assertEqual(env._root_puct_branch_observation_state_annotation_count, 1)


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
                    # Withdraw stays private: p2 never selects it before p1 snapshots the public
                    # branch point below.
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Withdraw")),)
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

    def test_public_materialization_constructs_a_fresh_sampled_branch_point(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),)
                ),
                "p2": pack_team(
                    # Withdraw is never selected in the source battle, so it
                    # must remain absent from p1's public materialization.
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Withdraw")),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            expected = source.observe("p1")

            # P-1: the live hidden-information battle must never send a Node
            # snapshot across the bridge. Audit the wire seam itself so this
            # remains true even if a later refactor bypasses request-event
            # helpers while extracting public state.
            source_commands: list[Mapping[str, Any]] = []
            original_send_command = source._send_command

            def record_source_command(payload: Mapping[str, Any]) -> None:
                source_commands.append(dict(payload))
                original_send_command(payload)

            with mock.patch.object(source, "_send_command", side_effect=record_source_command):
                materialization = source.public_materialization_state("p1")

            self.assertEqual(materialization.replay.requests, {})
            self.assertEqual(materialization.self_request["side"]["id"], "p1")
            self.assertFalse(hasattr(materialization, "bridge_snapshot"))
            self.assertEqual(
                source_commands,
                [],
                "public materialization must remain a bridge-free capture of the live battle",
            )
            public_payload = json.dumps(_public_materialization_payload(materialization), sort_keys=True)
            self.assertNotIn("withdraw", public_payload.lower())

            # A separate belief-sampled search world may retain a bridge
            # snapshot. This test invokes it only after direct materialization;
            # the companion live-environment test covers the start-override
            # permission gate that rejects snapshots for ordinary live games.
            search_commands: list[Mapping[str, Any]] = []
            original_search_send_command = search_env._send_command

            def record_search_command(payload: Mapping[str, Any]) -> None:
                search_commands.append(dict(payload))
                original_search_send_command(payload)

            with mock.patch.object(search_env, "_send_command", side_effect=record_search_command):
                search_env.materialize_public_world(
                    state=materialization,
                    start_override=start_override,
                    seed=7,
                )
                snapshot = search_env.snapshot_for_search()
                search_env.restore_search_snapshot(snapshot)
                self.assertTrue(search_env.release_search_snapshot(snapshot))
                actual = search_env.observe("p1")
                branch = search_env.step({"p1": 1, "p2": 1})

            self.assertIn("snapshot_search", {command.get("type") for command in search_commands})

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(branch.requested_players, ("p1", "p2"))

    def test_public_materialization_preserves_a_switched_active_pokemon(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),
                        FixturePokemon(species="Charmeleon", ability="Blaze", moves=("Ember", "Tackle")),
                    )
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Tackle")),
                        FixturePokemon(species="Wartortle", ability="Torrent", moves=("Water Gun", "Tackle")),
                    )
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 4, "p2": 0})  # p1 switches to Charmeleon.
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=7,
            )
            actual = search_env.observe("p1")
            branch = search_env.step({"p1": 0, "p2": 0})

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(branch.requested_players, ("p1", "p2"))

    def test_public_materialization_preserves_pending_wish(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Jirachi", ability="Serene Grace", moves=("Wish", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Tackle",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=29, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Jirachi uses Wish.
            materialization = source.public_materialization_state("p1")
            expected = source.step({"p1": 1, "p2": 0})

            self.assertEqual(materialization.replay.wish_set_turns, {"p1": 1})

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=29,
            )
            actual = search_env.step({"p1": 1, "p2": 0})

        self.assertEqual(
            actual.observations["p1"].metadata["self_team"][0]["condition"],
            expected.observations["p1"].metadata["self_team"][0]["condition"],
        )

    def test_public_materialization_preserves_leech_seed_residual(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Bulbasaur", ability="Overgrow", moves=("Leech Seed", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Splash", "Tackle")),)
                ),
            }
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=39, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Bulbasaur seeds Squirtle.
            materialization = source.public_materialization_state("p1")
            payload = _public_materialization_payload(materialization)
            self.assertEqual(materialization.replay.leech_seed_source_sides, {"p2": "p1"})
            self.assertEqual(payload["leechSeedSourceSides"], {"p2": "p1"})

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=39,
            )
            # Materialization starts a fresh simulator, so align post-boundary randomness before
            # comparing the next damage-plus-residual transition.
            source.reseed_simulator_rng(919)
            search_env.reseed_simulator_rng(919)
            source.step({"p1": 1, "p2": 0})
            search_env.step({"p1": 1, "p2": 0})
            source_hp = _active_hp_from_snapshot(source.snapshot(), "p2")
            search_hp = _active_hp_from_snapshot(search_env.snapshot(), "p2")

        self.assertEqual(search_hp, source_hp)

    def test_public_materialization_omits_expired_full_hp_wish(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Jirachi", ability="Serene Grace", moves=("Wish", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Splash",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source:
            source.reset_with_start_override(seed=31, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Jirachi uses Wish while at full HP.
            source.step({"p1": 1, "p2": 0})  # The landing does not emit a heal event.
            materialization = source.public_materialization_state("p1")

        # The protocol fold intentionally preserves the set turn for observation history, but
        # direct construction must not recreate a Wish that has already expired.
        self.assertEqual(materialization.replay.wish_set_turns, {"p1": 1})
        self.assertEqual(_public_materialization_payload(materialization)["wishSetTurns"], {})

    def test_public_materialization_preserves_wish_through_double_force_switch(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(species="Jirachi", ability="Serene Grace", moves=("Wish", "Tackle")),
                        FixturePokemon(species="Charmander", ability="Blaze", moves=("Tackle",)),
                    )
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(species="Snorlax", ability="Immunity", moves=("Explosion",)),
                        FixturePokemon(species="Magikarp", ability="Swift Swim", moves=("Splash",)),
                    )
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=35, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Wish is set before Snorlax's Explosion.
            materialization = source.public_materialization_state("p1")

            self.assertTrue(materialization.self_request["forceSwitch"][0])
            self.assertTrue(source._latest_requests["p2"]["forceSwitch"][0])
            self.assertEqual(_public_materialization_payload(materialization)["wishSetTurns"], {"p1": 1})
            expected_boundary = source.observe("p1")
            trajectory = BattleTrajectory(battle_id="double-force-switch", format_id="gen3randombattle", seed=35)
            prepared = prepare_direct_materialization_prefix(
                env=search_env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                start_override=start_override,
                public_materialization_state=materialization,
                expected_current_observation=expected_boundary,
            )
            self.assertIsNotNone(prepared)
            assert prepared is not None
            self.assertTrue(search_env.legal_actions("p2")[4])
            expected = source.step({"p1": 4, "p2": 4})
            search = puct_branch_search(
                env=search_env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=expected_boundary.legal_action_mask,
                opponent_actions={"p2": 4},
                value_fn=lambda _history: 0.0,
                action_priors=tuple(1.0 if index == 4 else 0.0 for index in range(9)),
                cpuct=1.0,
                root_visit_budget=1,
                start_override=start_override,
                expected_current_observation=expected_boundary,
                prepared_prefix=prepared,
            )
            release_prepared_replay_prefix(search_env, prepared)
            actual = search.candidates[0].value_candidate.branch.step_result

        self.assertEqual(
            actual.observations["p1"].metadata["self_active"]["condition"],
            expected.observations["p1"].metadata["self_active"]["condition"],
        )
        self.assertEqual(tuple(candidate.action_index for candidate in search.candidates), (4,))

    def test_public_materialization_preserves_three_member_actor_request_order(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),
                        FixturePokemon(species="Charmeleon", ability="Blaze", moves=("Ember", "Tackle")),
                        FixturePokemon(species="Charizard", ability="Blaze", moves=("Ember", "Tackle")),
                    )
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Tackle")),
                        FixturePokemon(species="Wartortle", ability="Torrent", moves=("Water Gun", "Tackle")),
                    )
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 5, "p2": 0})  # p1 switches to the third team member.
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")
            expected_branch = source.step({"p1": 4, "p2": 0})

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=7,
            )
            actual = search_env.observe("p1")
            self.assertEqual(search_env._latest_requests["p1"], materialization.self_request)
            branch = search_env.step({"p1": 4, "p2": 0})

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(
            branch.observations["p1"].categorical_ids,
            expected_branch.observations["p1"].categorical_ids,
        )
        self.assertEqual(
            branch.observations["p1"].legal_action_mask,
            expected_branch.observations["p1"].legal_action_mask,
        )

    def test_public_materialization_preserves_actor_known_pp_after_switching_out(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),
                        FixturePokemon(species="Charmeleon", ability="Blaze", moves=("Ember", "Tackle")),
                    )
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Tackle",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Charmander uses Ember.
            expected_pp = source._latest_requests["p1"]["active"][0]["moves"][0]["pp"]
            source.step({"p1": 4, "p2": 0})  # Charmander switches out.
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")

            self.assertIn("charmander", materialization.self_move_states)
            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=7,
            )
            actual = search_env.observe("p1")
            search_env.step({"p1": 4, "p2": 0})  # Switch back to Charmander.
            actual_pp = search_env._latest_requests["p1"]["active"][0]["moves"][0]["pp"]

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual_pp, expected_pp)

    def test_public_materialization_preserves_actor_known_trapped_constraint(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(species="Charmander", ability="Blaze", moves=("Tackle",)),
                        FixturePokemon(species="Charmeleon", ability="Blaze", moves=("Tackle",)),
                    )
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Diglett", ability="Arena Trap", moves=("Tackle",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")
            self.assertTrue(materialization.self_request["active"][0]["maybeTrapped"])

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=7,
            )
            actual = search_env.observe("p1")

        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(actual.numeric_features, expected.numeric_features)

    def test_public_materialization_preserves_force_switch_boundary(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(species="Magikarp", ability="Swift Swim", moves=("Tackle",), level=5),
                        FixturePokemon(species="Charmeleon", ability="Blaze", moves=("Tackle",)),
                    )
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Mewtwo", ability="Pressure", moves=("Psychic",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")
            self.assertTrue(materialization.self_request["forceSwitch"][0])
            expected_branch = source.step({"p1": 4})

            trajectory = BattleTrajectory(battle_id="force-switch", format_id="gen3randombattle", seed=7)
            prepared = prepare_direct_materialization_prefix(
                env=search_env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                start_override=start_override,
                public_materialization_state=materialization,
                expected_current_observation=expected,
            )
            self.assertIsNotNone(prepared)
            assert prepared is not None
            actual = search_env.observe("p1")
            search = puct_branch_search(
                env=search_env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=expected.legal_action_mask,
                opponent_actions={},
                value_fn=lambda _history: 0.0,
                action_priors=tuple(1.0 if index == 4 else 0.0 for index in range(9)),
                cpuct=1.0,
                root_visit_budget=1,
                start_override=start_override,
                expected_current_observation=expected,
                prepared_prefix=prepared,
            )
            release_prepared_replay_prefix(search_env, prepared)

        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(search.total_visits, 1)
        self.assertEqual(tuple(candidate.action_index for candidate in search.candidates), (4,))
        self.assertEqual(
            search.candidates[0].value_candidate.branch.step_result.observations["p1"].categorical_ids,
            expected_branch.observations["p1"].categorical_ids,
        )
        self.assertEqual(
            search.candidates[0].value_candidate.branch.step_result.observations["p1"].numeric_features,
            expected_branch.observations["p1"].numeric_features,
        )

    def test_public_materialization_preserves_spikes_layers(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Skarmory", ability="Keen Eye", moves=("Spikes", "Tackle")),)
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(species="Squirtle", ability="Torrent", moves=("Tackle",)),
                        FixturePokemon(species="Wartortle", ability="Torrent", moves=("Tackle",)),
                    )
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")
            expected_branch = source.step({"p1": 0, "p2": 4})

            self.assertEqual(materialization.replay.side_condition_counts["p2"], {"spikes": 1})

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=7,
            )
            actual = search_env.observe("p1")
            branch = search_env.step({"p1": 0, "p2": 4})

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(
            branch.observations["p1"].categorical_ids,
            expected_branch.observations["p1"].categorical_ids,
        )
        self.assertEqual(
            branch.observations["p1"].numeric_features,
            expected_branch.observations["p1"].numeric_features,
        )

    def test_public_materialization_preserves_move_weather(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Ludicolo", ability="Swift Swim", moves=("Rain Dance", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Tackle",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")

            self.assertEqual(materialization.replay.weather, "raindance")
            self.assertEqual(materialization.replay.weather_set_turn, 1)

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=7,
            )
            actual = search_env.observe("p1")
            for _ in range(4):
                search_env.step({"p1": 0, "p2": 0})
            final_materialization = search_env.public_materialization_state("p1")

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertIsNone(final_materialization.replay.weather)

    def test_public_materialization_preserves_permanent_ability_weather(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Tyranitar", ability="Sand Stream", moves=("Protect",)),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Protect",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")

            self.assertEqual(materialization.replay.weather, "sandstorm")
            self.assertTrue(materialization.replay.weather_from_ability)

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=7,
            )
            actual = search_env.observe("p1")
            for _ in range(4):
                search_env.step({"p1": 0, "p2": 0})
            final_materialization = search_env.public_materialization_state("p1")

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(final_materialization.replay.weather, "sandstorm")

    def test_public_materialization_preserves_reflect_duration(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Slowbro", ability="Oblivious", moves=("Reflect", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Tackle",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")

            self.assertEqual(materialization.replay.side_condition_counts["p1"], {"reflect": 1})
            self.assertEqual(materialization.replay.side_condition_set_turns["p1"], {"reflect": 1})

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=7,
            )
            actual = search_env.observe("p1")
            for _ in range(4):
                search_env.step({"p1": 0, "p2": 0})
            final_materialization = search_env.public_materialization_state("p1")

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertNotIn("reflect", final_materialization.replay.side_condition_counts["p1"])

    def test_public_materialization_preserves_toxic_residual_stage(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Bulbasaur", ability="Overgrow", moves=("Toxic", "Protect")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Harden",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=31, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Bulbasaur badly poisons Squirtle.
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")
            # Non-damaging choices isolate the deterministic toxic residual from future move RNG.
            expected_branch = source.step({"p1": 1, "p2": 0})

            self.assertEqual(materialization.replay.toxic_stage["p2"], 2)

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=31,
            )
            actual = search_env.observe("p1")
            branch = search_env.step({"p1": 1, "p2": 0})

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(branch.observations["p1"].categorical_ids, expected_branch.observations["p1"].categorical_ids)
        self.assertEqual(branch.observations["p1"].numeric_features, expected_branch.observations["p1"].numeric_features)

    def test_public_materialization_preserves_static_public_volatiles(self) -> None:
        config = integration_config()
        assert config is not None
        cases = (
            ("Charmander", "Blaze", "Focus Energy", "focusenergy"),
            ("Shuckle", "Sturdy", "Ingrain", "ingrain"),
            ("Mudkip", "Torrent", "Mud Sport", "mudsport"),
            ("Poliwag", "Water Absorb", "Water Sport", "watersport"),
        )

        for species, ability, setup_move, volatile in cases:
            with self.subTest(volatile=volatile):
                start_override = BattleStartOverride(
                    player_teams={
                        "p1": pack_team(
                            (FixturePokemon(
                                species=species,
                                ability=ability,
                                moves=(setup_move, "Protect"),
                            ),)
                        ),
                        "p2": pack_team(
                            (FixturePokemon(species="Ditto", ability="Limber", moves=("Harden",)),)
                        ),
                    },
                )
                with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
                    source.reset_with_start_override(seed=37, start_override=start_override)
                    source.step({"p1": 0, "p2": 0})
                    expected = source.observe("p1")
                    materialization = source.public_materialization_state("p1")
                    expected_branch = source.step({"p1": 1, "p2": 0})

                    self.assertIn(volatile, materialization.replay.volatiles["p1"])

                    search_env.materialize_public_world(
                        state=materialization,
                        start_override=start_override,
                        seed=37,
                    )
                    actual = search_env.observe("p1")
                    branch = search_env.step({"p1": 1, "p2": 0})

                self.assertEqual(actual.categorical_ids, expected.categorical_ids)
                self.assertEqual(actual.numeric_features, expected.numeric_features)
                self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
                self.assertEqual(
                    branch.observations["p1"].categorical_ids,
                    expected_branch.observations["p1"].categorical_ids,
                )
                self.assertEqual(
                    branch.observations["p1"].numeric_features,
                    expected_branch.observations["p1"].numeric_features,
                )

    def test_public_materialization_preserves_confirmed_trick_items(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(
                            species="Furret",
                            ability="Keen Eye",
                            item="Choice Band",
                            moves=("Trick", "Tackle"),
                        ),
                    )
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(
                            species="Alakazam",
                            ability="Synchronize",
                            item="Petaya Berry",
                            moves=("Harden", "Tackle"),
                        ),
                    )
                ),
            },
        )

        with (
            LocalShowdownEnv(config) as source,
            LocalShowdownEnv(config) as with_current_item,
            LocalShowdownEnv(config) as without_current_item,
        ):
            source.reset_with_start_override(seed=211, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Trick swaps Choice Band and Petaya Berry.
            materialization = source.public_materialization_state("p1")
            payload = _public_materialization_payload(materialization)

            self.assertEqual(payload["sides"]["p1"]["pokemon"][0]["currentItem"], "Petaya Berry")
            self.assertEqual(payload["sides"]["p2"]["pokemon"][0]["currentItem"], "Choice Band")
            source_items = {
                player: _active_item_state_from_snapshot(source.snapshot(), player)
                for player in ("p1", "p2")
            }

            with_current_item.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=211,
            )

            original_payload = local_showdown_module._public_materialization_payload

            def without_override_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
                stale_payload = original_payload(*args, **kwargs)
                for side in stale_payload["sides"].values():
                    for row in side["pokemon"]:
                        row.pop("currentItem", None)
                return stale_payload

            # Paired A/B: removing the protocol-confirmed override recreates the original
            # sampled items, while the production payload must match the live source exactly.
            with mock.patch.object(
                local_showdown_module,
                "_public_materialization_payload",
                side_effect=without_override_payload,
            ):
                without_current_item.materialize_public_world(
                    state=materialization,
                    start_override=start_override,
                    seed=211,
                )

            restored_items = {
                player: _active_item_state_from_snapshot(with_current_item.snapshot(), player)
                for player in ("p1", "p2")
            }
            stale_items = {
                player: _active_item_state_from_snapshot(without_current_item.snapshot(), player)
                for player in ("p1", "p2")
            }
            self.assertEqual(restored_items, source_items)
            self.assertNotEqual(stale_items, source_items)
            self.assertEqual(restored_items["p1"], ("petayaberry", "Choice Band"))
            self.assertEqual(restored_items["p2"], ("choiceband", "Petaya Berry"))

            source.reseed_simulator_rng(911)
            with_current_item.reseed_simulator_rng(911)
            without_current_item.reseed_simulator_rng(911)
            source.step({"p1": 1, "p2": 0})
            with_current_item.step({"p1": 1, "p2": 0})
            without_current_item.step({"p1": 1, "p2": 0})
            source_hp = _active_hp_from_snapshot(source.snapshot(), "p2")
            restored_hp = _active_hp_from_snapshot(with_current_item.snapshot(), "p2")
            stale_hp = _active_hp_from_snapshot(without_current_item.snapshot(), "p2")

        self.assertEqual(restored_hp, source_hp)
        self.assertLess(stale_hp, source_hp)

    def test_public_materialization_fails_closed_for_public_item_removal(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(
                            species="Sneasel",
                            ability="Keen Eye",
                            moves=("Knock Off", "Tackle"),
                        ),
                    )
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(
                            species="Alakazam",
                            ability="Synchronize",
                            item="Leftovers",
                            moves=("Harden",),
                        ),
                    )
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=223, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            materialization = source.public_materialization_state("p1")
            payload = _public_materialization_payload(materialization)

            self.assertIn(
                "item-state-removed:Alakazam",
                payload["sides"]["p2"]["materializationBlockers"],
            )
            with self.assertRaisesRegex(LocalShowdownError, "item-state-removed:Alakazam"):
                search_env.materialize_public_world(
                    state=materialization,
                    start_override=start_override,
                    seed=223,
                )

    def test_public_materialization_fails_closed_for_unconfirmed_item_mutation(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(
                            species="Sneasel",
                            ability="Keen Eye",
                            moves=("Thief", "Tackle"),
                        ),
                    )
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(
                            species="Alakazam",
                            ability="Synchronize",
                            item="Leftovers",
                            moves=("Harden",),
                        ),
                    )
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=227, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            materialization = source.public_materialization_state("p1")
            payload = _public_materialization_payload(materialization)

            self.assertIn(
                "item-state-unconfirmed:Sneasel",
                payload["sides"]["p1"]["materializationBlockers"],
            )
            self.assertIn(
                "item-state-unconfirmed:Alakazam",
                payload["sides"]["p2"]["materializationBlockers"],
            )
            with self.assertRaisesRegex(LocalShowdownError, "item-state-unconfirmed:Sneasel"):
                search_env.materialize_public_world(
                    state=materialization,
                    start_override=start_override,
                    seed=227,
                )

    def test_public_materialization_fails_closed_for_volatile_without_complete_public_state(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Pikachu", ability="Static", moves=("Substitute",)),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Ditto", ability="Limber", moves=("Harden",)),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=41, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            materialization = source.public_materialization_state("p1")

            self.assertIn("substitute", materialization.replay.volatiles["p1"])
            with self.assertRaisesRegex(LocalShowdownError, "volatile effect substitute"):
                search_env.materialize_public_world(
                    state=materialization,
                    start_override=start_override,
                    seed=41,
                )

    def test_public_materialization_preserves_static_volatile_mechanics(self) -> None:
        config = integration_config()
        assert config is not None
        cases = (
            (
                "flashfire",
                73,
                FixturePokemon(
                    species="Houndoom",
                    ability="Flash Fire",
                    moves=("Harden", "Ember"),
                ),
                FixturePokemon(
                    species="Charmander",
                    ability="Blaze",
                    moves=("Ember", "Harden"),
                ),
                {"p1": 0, "p2": 0},
                {"p1": 1, "p2": 1},
                "p2",
                "lower",
            ),
            (
                "focusenergy",
                2,
                FixturePokemon(species="Charmander", ability="Blaze", moves=("Focus Energy", "Tackle")),
                FixturePokemon(species="Ditto", ability="Limber", moves=("Harden",)),
                {"p1": 0, "p2": 0},
                {"p1": 1, "p2": 0},
                "p2",
                "lower",
            ),
            (
                "ingrain",
                37,
                FixturePokemon(species="Shuckle", ability="Sturdy", moves=("Ingrain", "Protect")),
                FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Harden")),
                {"p1": 0, "p2": 0},
                {"p1": 1, "p2": 1},
                "p1",
                "higher",
            ),
            (
                "mudsport",
                47,
                FixturePokemon(species="Mudkip", ability="Torrent", moves=("Mud Sport", "Tackle")),
                FixturePokemon(species="Pikachu", ability="Static", moves=("Harden", "Thunder Shock")),
                {"p1": 0, "p2": 0},
                {"p1": 1, "p2": 1},
                "p1",
                "higher",
            ),
            (
                "watersport",
                59,
                FixturePokemon(species="Poliwag", ability="Water Absorb", moves=("Water Sport", "Tackle")),
                FixturePokemon(species="Charmander", ability="Blaze", moves=("Harden", "Ember")),
                {"p1": 0, "p2": 0},
                {"p1": 1, "p2": 1},
                "p1",
                "higher",
            ),
        )

        for volatile, seed, p1, p2, setup_actions, branch_actions, affected_side, expected_direction in cases:
            with self.subTest(volatile=volatile):
                start_override = BattleStartOverride(
                    player_teams={"p1": pack_team((p1,)), "p2": pack_team((p2,))}
                )
                with (
                    LocalShowdownEnv(config) as source,
                    LocalShowdownEnv(config) as with_effect,
                    LocalShowdownEnv(config) as without_effect,
                ):
                    source.reset_with_start_override(seed=seed, start_override=start_override)
                    source.step(setup_actions)
                    expected = source.observe("p1")
                    materialization = source.public_materialization_state("p1")

                    self.assertIn(volatile, materialization.replay.volatiles["p1"])
                    without_volatile = replace(
                        materialization,
                        replay=replace(
                            materialization.replay,
                            volatiles={**materialization.replay.volatiles, "p1": ()},
                        ),
                    )

                    with_effect.materialize_public_world(
                        state=materialization,
                        start_override=start_override,
                        seed=seed,
                    )
                    without_effect.materialize_public_world(
                        state=without_volatile,
                        start_override=start_override,
                        seed=seed,
                    )
                    actual = with_effect.observe("p1")
                    # Re-seeding both worlds at the request boundary makes this a direct
                    # Showdown-equivalence assertion instead of a paired-only effect check.
                    source.reseed_simulator_rng(911)
                    with_effect.reseed_simulator_rng(911)
                    source.step(branch_actions)
                    with_effect.step(branch_actions)
                    without_effect.step(branch_actions)
                    source_hp = _active_hp_from_snapshot(source.snapshot(), affected_side)
                    with_hp = _active_hp_from_snapshot(with_effect.snapshot(), affected_side)
                    without_hp = _active_hp_from_snapshot(without_effect.snapshot(), affected_side)

                self.assertEqual(actual.categorical_ids, expected.categorical_ids)
                self.assertEqual(actual.numeric_features, expected.numeric_features)
                self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
                # The reconstructed world intentionally has a shorter public-event history,
                # so compare simulator behavior rather than history-derived feature rows.
                self.assertEqual(with_hp, source_hp)
                if expected_direction == "higher":
                    self.assertGreater(with_hp, without_hp)
                else:
                    self.assertLess(with_hp, without_hp)

    def test_public_materialization_preserves_opponent_static_public_volatile(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Ditto", ability="Limber", moves=("Harden",)),)
                ),
                "p2": pack_team(
                    (FixturePokemon(
                        species="Shuckle",
                        ability="Sturdy",
                        moves=("Ingrain", "Protect"),
                    ),)
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=43, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")
            expected_branch = source.step({"p1": 0, "p2": 1})

            self.assertIn("ingrain", materialization.replay.volatiles["p2"])

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=43,
            )
            actual = search_env.observe("p1")
            branch = search_env.step({"p1": 0, "p2": 1})

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(branch.observations["p1"].categorical_ids, expected_branch.observations["p1"].categorical_ids)
        self.assertEqual(branch.observations["p1"].numeric_features, expected_branch.observations["p1"].numeric_features)

    def test_public_materialization_preserves_ingrain_through_baton_pass(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(
                            species="Smeargle",
                            ability="Own Tempo",
                            moves=("Ingrain", "Baton Pass", "Protect"),
                        ),
                        FixturePokemon(
                            species="Bulbasaur",
                            ability="Overgrow",
                            moves=("Tackle", "Protect"),
                        ),
                    )
                ),
                "p2": pack_team((FixturePokemon(species="Ditto", ability="Limber", moves=("Harden",)),)),
            }
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=37, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Ingrain
            source.step({"p1": 1, "p2": 0})  # Baton Pass
            self.assertEqual(source.requested_players(), ("p1",))
            source.step({"p1": 4})  # Switch to Bulbasaur.
            expected = source.observe("p1")
            materialization = source.public_materialization_state("p1")

            self.assertIn("ingrain", materialization.replay.volatiles["p1"])
            self.assertEqual(materialization.replay.direct_materialization_blockers["p1"], ())
            self.assertTrue(source._latest_requests["p1"]["active"][0]["trapped"])

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=37,
            )
            actual = search_env.observe("p1")
            self.assertFalse(actual.legal_action_mask[4])

            source.reseed_simulator_rng(919)
            search_env.reseed_simulator_rng(919)
            source.step({"p1": 1, "p2": 0})
            search_env.step({"p1": 1, "p2": 0})
            source_hp = _active_hp_from_snapshot(source.snapshot(), "p1")
            search_hp = _active_hp_from_snapshot(search_env.snapshot(), "p1")

        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(search_hp, source_hp)

    def test_public_materialization_preserves_pending_baton_pass_at_forced_switch(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(
                            species="Smeargle",
                            ability="Own Tempo",
                            moves=("Swords Dance", "Baton Pass", "Protect"),
                        ),
                        FixturePokemon(
                            species="Bulbasaur",
                            ability="Overgrow",
                            moves=("Tackle", "Protect"),
                        ),
                    )
                ),
                "p2": pack_team((FixturePokemon(species="Ditto", ability="Limber", moves=("Harden",)),)),
            }
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=73, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Swords Dance
            source.step({"p1": 1, "p2": 0})  # Baton Pass, then p1 must select a switch.
            self.assertEqual(source.requested_players(), ("p1",))
            materialization = source.public_materialization_state("p1")
            self.assertEqual(materialization.replay.pending_baton_pass, ("p1",))
            self.assertEqual(
                _public_materialization_payload(materialization)["pendingBatonPassSides"], ["p1"]
            )
            with self.assertRaisesRegex(ValueError, "invalid deferred opponent action"):
                _public_materialization_payload(
                    materialization,
                    deferred_opponent_actions={"p2": 4},
                )

            expected_boundary = source.observe("p1")
            trajectory = BattleTrajectory(battle_id="pending-baton-pass", format_id="gen3randombattle", seed=73)
            prepared = prepare_direct_materialization_prefix(
                env=search_env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                start_override=start_override,
                public_materialization_state=materialization,
                deferred_opponent_actions={"p2": 0},
                expected_current_observation=expected_boundary,
            )
            self.assertIsNotNone(prepared)
            assert prepared is not None
            self.assertEqual(search_env.requested_players(), ("p1",))

            source.step({"p1": 4})
            search = puct_branch_search(
                env=search_env,
                trajectory=trajectory,
                player_id="p1",
                prefix_decision_round_count=0,
                legal_action_mask=expected_boundary.legal_action_mask,
                opponent_actions={},
                value_fn=lambda _history: 0.0,
                action_priors=tuple(1.0 if index == 4 else 0.0 for index in range(9)),
                cpuct=1.0,
                root_visit_budget=1,
                start_override=start_override,
                expected_current_observation=expected_boundary,
                prepared_prefix=prepared,
            )
            release_prepared_replay_prefix(search_env, prepared)
            expected = source.observe("p1")
            actual = search.candidates[0].value_candidate.branch.step_result.observations["p1"]
            ordinary_boundary = source.public_materialization_state("p1")
            stale_actor_boundary = replace(
                ordinary_boundary,
                replay=replace(ordinary_boundary.replay, pending_baton_pass=("p1",)),
            )
            stale_opponent_boundary = replace(
                ordinary_boundary,
                replay=replace(ordinary_boundary.replay, pending_baton_pass=("p2",)),
            )

        # The opposing action is hidden at the forced-switch boundary. The direct world samples
        # it into the restored queue, so both the pass and the interrupted turn resolve normally.
        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(tuple(candidate.action_index for candidate in search.candidates), (4,))
        self.assertEqual(source._parser.snapshot().boosts["p1"], {"atk": 2})
        self.assertEqual(search_env._parser.snapshot().boosts["p1"], {"atk": 2})
        self.assertEqual(_public_materialization_payload(stale_actor_boundary)["pendingBatonPassSides"], [])
        self.assertEqual(_public_materialization_payload(stale_opponent_boundary)["pendingBatonPassSides"], [])

    def test_public_materialization_samples_deferred_baton_pass_action_without_private_request(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(
                            species="Smeargle",
                            ability="Own Tempo",
                            moves=("Swords Dance", "Baton Pass", "Protect"),
                        ),
                        FixturePokemon(
                            species="Bulbasaur",
                            ability="Overgrow",
                            moves=("Tackle", "Protect"),
                        ),
                    )
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Ditto", ability="Limber", moves=("Harden", "Protect")),)
                ),
            }
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=79, start_override=start_override)
            source.step({"p1": 0, "p2": 0})  # Swords Dance versus the private Harden request.
            source.step({"p1": 1, "p2": 0})  # Baton Pass interrupts before Harden resolves.
            materialization = source.public_materialization_state("p1")
            payload = _public_materialization_payload(materialization)

            # Neither the pending source move nor an alternative must be copied from p2's private
            # request. The direct world receives only an action index sampled by the planner.
            self.assertIn("harden", json.dumps(source._latest_requests["p2"]).casefold())
            self.assertIn("protect", json.dumps(source._latest_requests["p2"]).casefold())
            serialized_payload = json.dumps(payload, sort_keys=True).casefold()
            self.assertNotIn("harden", serialized_payload)
            self.assertNotIn("moves", payload["sides"]["p2"]["pokemon"][0])
            self.assertEqual(payload["deferredOpponentActions"], {})

            # Action one is Protect, not the live battle's previously chosen Harden. This proves
            # the restored queue follows the predictor's sampled world rather than the live queue.
            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=79,
                deferred_opponent_actions={"p2": 1},
            )
            search_env.step({"p1": 4})
            protocol = "\n".join(search_env.protocol_lines)

            # A hidden-mode planner can rank slots that do not exist in the sampled world. Its
            # public priors are conditioned on sampled legal slots without reading a source
            # battle request, so the best available slot is selected without duplicate worlds.
            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=79,
                deferred_opponent_action_priors={"p2": (0.10, 0.90, 0.30, 0.40)},
            )
            search_env.step({"p1": 4})
            conditioned_protocol = "\n".join(search_env.protocol_lines)

        self.assertIn("|move|p2a: Ditto|Protect|", protocol)
        self.assertNotIn("|move|p2a: Ditto|Harden|", protocol)
        self.assertIn("|move|p2a: Ditto|Protect|", conditioned_protocol)
        self.assertNotIn("|move|p2a: Ditto|Harden|", conditioned_protocol)
        self.assertEqual(protocol.count("|upkeep"), 1)
        self.assertEqual(conditioned_protocol.count("|upkeep"), 1)

    def test_public_materialization_fails_closed_for_baton_passed_substitute(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(
                            species="Smeargle",
                            ability="Own Tempo",
                            moves=("Substitute", "Baton Pass"),
                        ),
                        FixturePokemon(
                            species="Bulbasaur",
                            ability="Overgrow",
                            moves=("Tackle",),
                        ),
                    )
                ),
                "p2": pack_team((FixturePokemon(species="Ditto", ability="Limber", moves=("Harden",)),)),
            }
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=41, start_override=start_override)
            source.step({"p1": 0, "p2": 0})
            source.step({"p1": 1, "p2": 0})
            source.step({"p1": 4})
            materialization = source.public_materialization_state("p1")

            self.assertIn("substitute", materialization.replay.volatiles["p1"])
            self.assertEqual(
                materialization.replay.direct_materialization_blockers["p1"],
                ("baton-pass:substitute",),
            )
            with self.assertRaisesRegex(LocalShowdownError, "baton-pass:substitute"):
                search_env.materialize_public_world(
                    state=materialization,
                    start_override=start_override,
                    seed=41,
                )

    def test_public_materialization_fails_closed_when_actor_pp_history_is_unavailable(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (
                        FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),
                        FixturePokemon(species="Charmeleon", ability="Blaze", moves=("Ember", "Tackle")),
                    )
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Tackle")),
                        FixturePokemon(species="Wartortle", ability="Torrent", moves=("Water Gun", "Tackle")),
                    )
                ),
            },
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=7, start_override=start_override)
            source.step({"p1": 0, "p2": 1})  # Charmander spends PP before switching out.
            source.step({"p1": 4, "p2": 1})
            materialization = source.public_materialization_state("p1")
            # The runtime retains actor-owned history. Confirm the safety check remains closed
            # when that player-known state is unavailable rather than inventing benched PP.
            materialization = replace(materialization, self_move_states={})

            with self.assertRaisesRegex(LocalShowdownError, "spent PP for a benched"):
                search_env.materialize_public_world(
                    state=materialization,
                    start_override=start_override,
                    seed=7,
                )

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

    def test_search_snapshot_handle_replays_same_branch_without_exposing_simulator_state(self) -> None:
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
            snapshot = env.snapshot_for_search()
            self.assertIsInstance(snapshot.bridge_snapshot.get("snapshot_id"), str)
            self.assertNotIn("battle", snapshot.bridge_snapshot)
            prefix_len = len(snapshot.protocol_lines)
            env.step({"p1": 0, "p2": 1})
            expected_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])

            env.reset_with_start_override(seed=19, start_override=start_override)
            env.restore_search_snapshot(snapshot)
            self.assertEqual(env.requested_players(), ("p1", "p2"))
            env.step({"p1": 0, "p2": 1})
            restored_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])

            # A second restore of the same handle must start from an independent engine clone,
            # not from the state mutated by the prior restored branch.
            env.restore_search_snapshot(snapshot)
            env.step({"p1": 1, "p2": 0})
            env.restore_search_snapshot(snapshot)
            env.step({"p1": 0, "p2": 1})
            repeated_restored_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])

        self.assertEqual(restored_suffix, expected_suffix)
        self.assertEqual(repeated_restored_suffix, expected_suffix)

    def test_search_snapshot_handle_fuses_restore_and_branch_step(self) -> None:
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
            snapshot = env.snapshot_for_search()
            prefix_len = len(snapshot.protocol_lines)

            before_explicit = env.root_puct_bridge_timing_snapshot()
            env.restore_search_snapshot(snapshot)
            explicit_result = env.step({"p1": 0, "p2": 1})
            explicit_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])
            after_explicit = env.root_puct_bridge_timing_snapshot()

            before_fused = env.root_puct_bridge_timing_snapshot()
            before_fused_branch = env.root_puct_branch_step_timing_snapshot()
            fused_result = env.step_from_search_snapshot(snapshot, {"p1": 0, "p2": 1})
            fused_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])
            after_fused = env.root_puct_bridge_timing_snapshot()
            after_fused_branch = env.root_puct_branch_step_timing_snapshot()

            # The retained handle remains an immutable branch point after a fused visit.
            env.step_from_search_snapshot(snapshot, {"p1": 1, "p2": 0})
            env.step_from_search_snapshot(snapshot, {"p1": 0, "p2": 1})
            repeated_fused_suffix = _without_timestamp_lines(env.protocol_lines[prefix_len:])

        self.assertEqual(fused_suffix, explicit_suffix)
        self.assertEqual(repeated_fused_suffix, explicit_suffix)
        self.assertEqual(fused_result.requested_players, explicit_result.requested_players)
        self.assertEqual(fused_result.rewards, explicit_result.rewards)
        self.assertEqual(fused_result.terminal, explicit_result.terminal)
        self.assertEqual(after_explicit["bridge_round_trip_count"] - before_explicit["bridge_round_trip_count"], 2)
        self.assertEqual(after_fused["bridge_round_trip_count"] - before_fused["bridge_round_trip_count"], 1)
        self.assertEqual(
            after_fused["bridge_node_processing_count"] - before_fused["bridge_node_processing_count"],
            1,
        )
        for key in (
            "branch_local_state_restore_count",
            "branch_choice_encoding_count",
            "branch_bridge_round_trip_count",
            "branch_bridge_node_processing_count",
            "branch_result_projection_count",
        ):
            self.assertEqual(after_fused_branch[key] - before_fused_branch[key], 1)
        self.assertEqual(
            after_fused_branch["branch_observation_projection_count"]
            - before_fused_branch["branch_observation_projection_count"],
            2,
        )
        for key in (
            "branch_observation_state_normalization_count",
            "branch_observation_incremental_sync_count",
            "branch_observation_replay_snapshot_count",
            "branch_observation_player_state_normalization_count",
            "branch_observation_state_annotation_count",
            "branch_observation_encoding_count",
            "branch_belief_overlay_projection_count",
        ):
            self.assertEqual(after_fused_branch[key] - before_fused_branch[key], 2)
        for key in (
            "branch_local_state_restore_seconds",
            "branch_choice_encoding_seconds",
            "branch_bridge_round_trip_seconds",
            "branch_bridge_node_processing_seconds",
            "branch_result_projection_seconds",
            "branch_observation_projection_seconds",
            "branch_observation_state_normalization_seconds",
            "branch_observation_incremental_sync_seconds",
            "branch_observation_replay_snapshot_seconds",
            "branch_observation_player_state_normalization_seconds",
            "branch_observation_state_annotation_seconds",
            "branch_observation_encoding_seconds",
            "branch_belief_overlay_projection_seconds",
        ):
            self.assertGreaterEqual(after_fused_branch[key] - before_fused_branch[key], 0.0)
        nested_state_normalization_seconds = sum(
            after_fused_branch[key] - before_fused_branch[key]
            for key in (
                "branch_observation_incremental_sync_seconds",
                "branch_observation_replay_snapshot_seconds",
                "branch_observation_player_state_normalization_seconds",
                "branch_observation_state_annotation_seconds",
            )
        )
        parent_state_normalization_seconds = (
            after_fused_branch["branch_observation_state_normalization_seconds"]
            - before_fused_branch["branch_observation_state_normalization_seconds"]
        )
        # The outer observation timer must contain every nested attribution timer.
        self.assertGreaterEqual(
            parent_state_normalization_seconds + 1e-9,
            nested_state_normalization_seconds,
        )

    def test_search_snapshot_fast_path_reuses_choices_and_limits_zero_rollout_view(self) -> None:
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
            }
        )

        with LocalShowdownEnv(config) as env:
            env.reset_with_start_override(seed=23, start_override=start_override)
            snapshot = env.snapshot_for_search()
            self.assertEqual(set(snapshot.search_choice_cache), {"p1", "p2"})
            self.assertIn(0, snapshot.search_choice_cache["p1"])
            self.assertIn(1, snapshot.search_choice_cache["p2"])

            with mock.patch.object(env, "_choices_for_actions", side_effect=AssertionError("cache miss")):
                result = env.step_from_search_snapshot_for_player(
                    snapshot,
                    {"p1": 0, "p2": 1},
                    observation_player="p1",
                )

            with self.assertRaisesRegex(ValueError, "p1: action_index 8 is not legal"):
                env.step_from_search_snapshot_for_player(
                    snapshot,
                    {"p1": 8, "p2": 1},
                    observation_player="p1",
                )
            # Integral floats compare equal to integer dictionary keys, but the
            # uncached translator rejects them while indexing the legal mask.
            # Keep the fast path on that exact validation behavior.
            with self.assertRaises(TypeError):
                env._cached_search_choices(snapshot, {"p1": 1.0, "p2": 1})

        self.assertEqual(set(result.observations), {"p1"})
        self.assertEqual(result.requested_players, ("p1", "p2"))

    def test_search_snapshot_restores_independent_annotation_trackers(self) -> None:
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
                    (
                        FixturePokemon(
                            species="Tauros",
                            ability="Intimidate",
                            moves=("Double-Edge", "Earthquake"),
                            level=76,
                        ),
                    )
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(
                            species="Snorlax",
                            ability="Immunity",
                            moves=("Body Slam", "Earthquake"),
                            level=71,
                        ),
                    )
                ),
            },
            # The custom-game shell makes the sampled-world snapshot controllable;
            # the observation format keeps the source-backed Gen 3 set universe
            # active for the annotation evidence this regression exercises.
            observation_format_id="gen3randombattle",
        )

        with LocalShowdownEnv(config) as env:
            env.reset_with_start_override(seed=23, start_override=start_override)
            # Build a real annotation prefix before capturing the search snapshot.
            # Tauros/Snorlax are source-backed sets, and this turn yields a residual
            # for each perspective plus one assessed investment strike per tracker.
            env.step({"p1": 0, "p2": 0})
            snapshot = env.snapshot_for_search()
            annotation_cache = snapshot.annotation_cache
            self.assertIsNotNone(annotation_cache)
            assert annotation_cache is not None
            self.assertEqual(set(annotation_cache.tier2_trackers), {"p1", "p2"})
            self.assertEqual(set(annotation_cache.investment_trackers), {"p1", "p2"})
            self.assertTrue(
                all(tracker._residuals for tracker in annotation_cache.tier2_trackers.values())
            )
            self.assertTrue(
                all(tracker._state.strikes for tracker in annotation_cache.investment_trackers.values())
            )

            def tracker_state(
                tier2_trackers: Mapping[str, Any], investment_trackers: Mapping[str, Any]
            ) -> tuple[dict[str, tuple[object, ...]], dict[str, tuple[object, ...]]]:
                return (
                    {
                        player: (
                            tracker._fold._processed,
                            tracker._assessed_until,
                            dict(tracker._residuals),
                            {key: tuple(turns) for key, turns in tracker._cb_turns.items()},
                            set(tracker._cb_non_ko),
                            set(tracker._cb_bit_indices),
                        )
                        for player, tracker in tier2_trackers.items()
                    },
                    {
                        player: (
                            tracker._fold._processed,
                            tracker._assessed_until,
                            tuple(tracker._state.strikes),
                            dict(tracker._state.token_codes),
                            {
                                key: (
                                    list(ledger.hp.pins),
                                    ledger.hp.blocked,
                                    ledger.hp.concluded_value,
                                    ledger.hp.concluded_class,
                                    ledger.hp.concluded_turns,
                                    {
                                        stat: (
                                            list(axis.pins),
                                            axis.blocked,
                                            axis.concluded_value,
                                            axis.concluded_class,
                                            axis.concluded_turns,
                                        )
                                        for stat, axis in ledger.defense.items()
                                    },
                                )
                                for key, ledger in tracker._state.ledgers.items()
                            },
                        )
                        for player, tracker in investment_trackers.items()
                    },
                )

            prefix_state = tracker_state(
                annotation_cache.tier2_trackers,
                annotation_cache.investment_trackers,
            )

            # Branch A grows both trackers beyond the non-empty prefix.
            first_result = env.step_from_search_snapshot(snapshot, {"p1": 1, "p2": 1})
            self.assertIsNot(env._tier2_trackers["p1"], annotation_cache.tier2_trackers["p1"])
            self.assertIsNot(
                env._investment_trackers["p1"], annotation_cache.investment_trackers["p1"]
            )
            self.assertGreater(
                env._tier2_trackers["p1"]._fold._processed,
                annotation_cache.tier2_trackers["p1"]._fold._processed,
            )
            self.assertGreater(
                len(env._tier2_trackers["p1"]._residuals),
                len(annotation_cache.tier2_trackers["p1"]._residuals),
            )
            self.assertGreater(
                len(env._investment_trackers["p1"]._state.strikes),
                len(annotation_cache.investment_trackers["p1"]._state.strikes),
            )
            self.assertEqual(
                tracker_state(
                    annotation_cache.tier2_trackers,
                    annotation_cache.investment_trackers,
                ),
                prefix_state,
            )

            env.restore_search_snapshot(snapshot)
            self.assertIsNot(env._tier2_trackers["p1"], annotation_cache.tier2_trackers["p1"])
            self.assertIsNot(
                env._investment_trackers["p1"], annotation_cache.investment_trackers["p1"]
            )
            self.assertEqual(
                env._tier2_trackers["p1"]._assessed_until,
                annotation_cache.tier2_trackers["p1"]._assessed_until,
            )
            self.assertEqual(
                env._investment_trackers["p1"]._assessed_until,
                annotation_cache.investment_trackers["p1"]._assessed_until,
            )
            self.assertEqual(
                tracker_state(env._tier2_trackers, env._investment_trackers),
                prefix_state,
            )

            # Branch B takes a distinct suffix; it must also leave the captured
            # prefix unmodified. Replaying branch A then proves one sibling cannot
            # leak its annotation state into another.
            env.step_from_search_snapshot(snapshot, {"p1": 1, "p2": 0})
            self.assertEqual(
                tracker_state(
                    annotation_cache.tier2_trackers,
                    annotation_cache.investment_trackers,
                ),
                prefix_state,
            )
            replayed_result = env.step_from_search_snapshot(snapshot, {"p1": 1, "p2": 1})

        self.assertEqual(first_result.observations, replayed_result.observations)

    def test_root_puct_bridge_timing_tracks_completed_search_commands(self) -> None:
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
            before = env.root_puct_bridge_timing_snapshot()
            snapshot = env.snapshot_for_search()
            env.step({"p1": 0, "p2": 1})
            env.restore_search_snapshot(snapshot)
            after = env.root_puct_bridge_timing_snapshot()

        # Snapshot, one branch choice, and restore each complete one bridge
        # request/response exchange and include a Node processing measurement.
        self.assertEqual(after["bridge_round_trip_count"] - before["bridge_round_trip_count"], 3)
        self.assertEqual(
            after["bridge_node_processing_count"] - before["bridge_node_processing_count"],
            3,
        )
        self.assertGreater(after["bridge_round_trip_seconds"] - before["bridge_round_trip_seconds"], 0.0)
        self.assertGreaterEqual(
            after["bridge_node_processing_seconds"] - before["bridge_node_processing_seconds"],
            0.0,
        )
        self.assertLessEqual(
            after["bridge_node_processing_seconds"] - before["bridge_node_processing_seconds"],
            after["bridge_round_trip_seconds"] - before["bridge_round_trip_seconds"],
        )

    def test_search_snapshot_handle_restores_direct_materialized_public_state(self) -> None:
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
            }
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as search_env:
            source.reset_with_start_override(seed=17, start_override=start_override)
            source.step({"p1": 0, "p2": 1})
            materialization = source.public_materialization_state("p1")

            search_env.materialize_public_world(
                state=materialization,
                start_override=start_override,
                seed=17,
            )
            expected = search_env.observe("p1")
            expected_replay = search_env._parser.snapshot()
            expected_belief = expected.metadata["belief_view"]
            snapshot = search_env.snapshot_for_search()

            search_env.step({"p1": 0, "p2": 1})
            expected_next = search_env.observe("p1")
            search_env.reset_with_start_override(seed=19, start_override=start_override)
            search_env.restore_search_snapshot(snapshot)
            actual = search_env.observe("p1")
            actual_replay = search_env._parser.snapshot()
            search_env.step({"p1": 0, "p2": 1})
            actual_next = search_env.observe("p1")

        self.assertEqual(actual_replay, expected_replay)
        self.assertEqual(actual.metadata["belief_view"], expected_belief)
        self.assertEqual(actual.categorical_ids, expected.categorical_ids)
        self.assertEqual(actual.numeric_features, expected.numeric_features)
        self.assertEqual(actual.legal_action_mask, expected.legal_action_mask)
        self.assertEqual(actual_next.metadata["belief_view"], expected_next.metadata["belief_view"])
        self.assertEqual(actual_next.categorical_ids, expected_next.categorical_ids)
        self.assertEqual(actual_next.numeric_features, expected_next.numeric_features)
        self.assertEqual(actual_next.legal_action_mask, expected_next.legal_action_mask)

    def test_search_snapshot_handle_rejects_live_rollout(self) -> None:
        config = integration_config()
        assert config is not None
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team((FixturePokemon(species="Charmander", moves=("Ember",)),)),
                "p2": pack_team((FixturePokemon(species="Squirtle", moves=("Water Gun",)),)),
            },
        )

        with LocalShowdownEnv(config) as env:
            env.reset(seed=17)
            with self.assertRaisesRegex(LocalShowdownError, "belief-sampled start override"):
                env.snapshot_for_search()

            env.reset_with_start_override(seed=19, start_override=start_override)
            snapshot = env.snapshot_for_search()
            env.reset(seed=23)
            with self.assertRaisesRegex(LocalShowdownError, "belief-sampled start override"):
                env.restore_search_snapshot(snapshot)
            with self.assertRaisesRegex(LocalShowdownError, "belief-sampled start override"):
                env.step_from_search_snapshot(snapshot, {"p1": 0, "p2": 0})

            env.reset_with_start_override(seed=29, start_override=start_override)
            self.assertTrue(env.release_search_snapshot(snapshot))
            self.assertFalse(env.release_search_snapshot(snapshot))
            with self.assertRaisesRegex(LocalShowdownError, "Unknown search snapshot"):
                env.restore_search_snapshot(snapshot)
            with self.assertRaisesRegex(LocalShowdownError, "Unknown search snapshot"):
                env.step_from_search_snapshot(snapshot, {"p1": 0, "p2": 0})

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

    def test_snapshot_restore_preserves_investment_trackers(self) -> None:
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
                    (
                        FixturePokemon(
                            species="Tauros",
                            ability="Intimidate",
                            moves=("Double-Edge", "Earthquake"),
                            level=76,
                        ),
                    )
                ),
                "p2": pack_team(
                    (
                        FixturePokemon(
                            species="Snorlax",
                            ability="Immunity",
                            moves=("Body Slam", "Earthquake"),
                            level=71,
                        ),
                    )
                ),
            },
            observation_format_id="gen3randombattle",
        )

        with LocalShowdownEnv(config) as env:
            env.reset_with_start_override(seed=19, start_override=start_override)
            env.step({"p1": 0, "p2": 0})
            env.observe("p1")
            self.assertIn("p1", env._investment_trackers)
            self.assertTrue(env._tier2_trackers["p1"]._residuals)
            self.assertTrue(env._investment_trackers["p1"]._state.strikes)
            snapshot = env.snapshot()
            before_restore = env.observe("p1")
            env.step({"p1": 0, "p2": 0})
            self.assertIn("p1", env._investment_trackers)

            env.restore(snapshot)
            self.assertIn("p1", env._investment_trackers)
            self.assertEqual(env.observe("p1"), before_restore)

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
