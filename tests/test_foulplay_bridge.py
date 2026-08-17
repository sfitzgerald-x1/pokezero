from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.collection import read_rollout_records
from pokezero.foulplay_bridge import (
    FOULPLAY_THINK_MAX_COVERAGE_GAP,
    FOULPLAY_THINK_MIN_MEASURED_DECISIONS,
    FOULPLAY_THINK_MIN_STRATUM_DECISIONS,
    FOULPLAY_THINK_SCHEMA_VERSION,
    _FOULPLAY_THINK_LINE_CAP,
    _FOULPLAY_THINK_SETTLE_GIVEUP_DECISIONS,
    _FOULPLAY_THINK_SETTLE_SECONDS,
    _FoulPlayThinkClock,
    _ProcessLogBuffer,
    _foulplay_think_aggregate,
    _foulplay_think_observation,
    _foulplay_think_work_from_log_lines,
    _probe_foulplay_start_method,
    _wait_for_foulplay_choice_or_exit,
    compare_foulplay_think,
    foulplay_think_reading_status,
    ControlledFoulPlayBenchmarkResult,
    ControlledFoulPlayComparisonResult,
    ControlledFoulPlayConfig,
    ControlledFoulPlayGameResult,
    FoulPlayProcessExitError,
    _ControlledBattleState,
    _FoulPlayWebsocketServer,
    _choice_body_from_outgoing_message,
    _controlled_foulplay_comparison_progress_callback,
    _config_from_args,
    _capture_resolved_public_action_round,
    _build_policy,
    _foulplay_command,
    _foulplay_env,
    _handle_decision_boundary,
    _handle_stream_event,
    _line_for_foulplay,
    _line_chunks_safe_for_foulplay,
    _observation_with_search_metadata,
    _public_materialization_state,
    _player_state,
    _root_puct_prior_action_change_details,
    _root_puct_timing_from_metadata,
    _run_single_game,
    _requested_legal_action_masks_for_context,
    _is_terminal_protocol_line,
    _split_outgoing_showdown_message,
    _terminal_line_for_foulplay,
    _write_json,
    async_comparison_main,
    async_main,
    build_arg_parser,
    build_comparison_arg_parser,
    capture_controlled_foulplay_rollouts,
    run_controlled_foulplay_comparison,
    run_controlled_foulplay_benchmark,
)
from pokezero.env import TerminalState
from pokezero.foulplay_capture import async_main as async_capture_main
from pokezero.foulplay_capture import build_capture_arg_parser
from pokezero.public_decision_corpus import load_public_decision_corpus
from pokezero.neural_policy import TransformerTrainingConfig, require_torch, torch_available
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyDecision
from pokezero.search import RootPUCTSearchTiming
from pokezero.showdown import V2_2_REPLAY_OBSERVATION_SPEC
from pokezero.trajectory import BattleTrajectory, TrajectoryStep
from pokezero.value_calibration import evaluate_value_calibration


class FoulPlayBridgeTest(unittest.TestCase):
    def test_capture_parser_forces_raw_policy_mode(self) -> None:
        parser = build_capture_arg_parser()
        args = parser.parse_args(["--checkpoint", "checkpoint.pt", "--out", "pool.jsonl"])

        self.assertEqual(args.policy_mode, "raw")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--checkpoint", "checkpoint.pt", "--out", "pool.jsonl", "--policy-mode", "root-puct"]
            )

    def test_config_exposes_the_opposing_foulplay_seat(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            pokezero_player="p2",
        )

        self.assertEqual(config.foulplay_player, "p1")

    def test_stream_forwards_requests_for_the_configured_foulplay_seat(self) -> None:
        class Server:
            def __init__(self) -> None:
                self.messages: list[tuple[str, list[str]]] = []

            async def send_room_lines(self, battle_id: str, lines: list[str]) -> None:
                self.messages.append((battle_id, lines))

        state = _ControlledBattleState(battle_id="controlled-7", seed=7, format_id="gen3randombattle")
        server = Server()
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            pokezero_player="p2",
        )

        asyncio.run(
            _handle_stream_event(
                state,
                server,  # type: ignore[arg-type]
                {"stream": "p1", "lines": ['|request|{"side":{"id":"p1"}}']},
                config=config,
            )
        )

        self.assertIn("p1", state.request_lines)
        self.assertEqual(server.messages[0][0], "controlled-7")
        self.assertIn('"rqid":1', server.messages[0][1][0])

    def test_stream_never_forwards_the_configured_pokezero_seat(self) -> None:
        class Server:
            def __init__(self) -> None:
                self.messages: list[tuple[str, list[str]]] = []

            async def send_room_lines(self, battle_id: str, lines: list[str]) -> None:
                self.messages.append((battle_id, lines))

        state = _ControlledBattleState(battle_id="controlled-7", seed=7, format_id="gen3randombattle")
        server = Server()
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            pokezero_player="p2",
        )

        asyncio.run(
            _handle_stream_event(
                state,
                server,  # type: ignore[arg-type]
                {"stream": "p2", "lines": ['|request|{"side":{"id":"p2"}}']},
                config=config,
            )
        )

        self.assertIn("p2", state.request_lines)
        self.assertEqual(server.messages, [])

    def test_decision_boundary_selects_and_records_the_configured_pokezero_seat(self) -> None:
        class PlayerState:
            def __init__(self, slot: str) -> None:
                self.slot = slot

        class Bridge:
            def __init__(self) -> None:
                self.messages: list[dict] = []

            async def send(self, payload: dict) -> None:
                self.messages.append(payload)

        def observation(slot: str) -> PokeZeroObservationV0:
            return PokeZeroObservationV0(
                categorical_ids=(),
                numeric_features=(),
                token_type_ids=(),
                attention_mask=(),
                legal_action_mask=tuple(index == 0 for index in range(ACTION_COUNT)),
                metadata={"slot": slot},
            )

        state = _ControlledBattleState(
            battle_id="controlled-7",
            seed=7,
            format_id="gen3randombattle",
            trajectory=BattleTrajectory(battle_id="controlled-7", format_id="gen3randombattle", seed=7),
        )
        bridge = Bridge()
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            pokezero_player="p2",
        )
        contexts = []

        def select(_policy, _observation, context, *, seed):
            self.assertEqual(seed, 7)
            contexts.append(context)
            # A checkpoint may legitimately use the evaluator's display id. Seat telemetry must
            # therefore use controller provenance, never classify decisions by this id.
            return PolicyDecision(action_index=0, policy_id="foul-play")

        async def foulplay_choice(**_kwargs) -> str:
            return "move 1"

        def decode(player_state, _choice) -> int:
            self.assertEqual(player_state.slot, "p1")
            return 0

        with (
            patch("pokezero.foulplay_bridge._player_state", side_effect=lambda _state, slot, **_kwargs: PlayerState(slot)),
            patch(
                "pokezero.foulplay_bridge.observation_from_player_state",
                side_effect=lambda player_state, **_kwargs: observation(player_state.slot),
            ),
            patch("pokezero.foulplay_bridge._observation_with_search_metadata", side_effect=lambda value, _state: value),
            patch("pokezero.foulplay_bridge._select_policy_decision", side_effect=select),
            patch(
                "pokezero.foulplay_bridge.showdown_choice_for_action",
                side_effect=lambda player_state, action: f"{player_state.slot}:{action}",
            ),
            patch("pokezero.foulplay_bridge.time.perf_counter", side_effect=(10.0, 12.5)),
            patch("pokezero.foulplay_bridge._wait_for_foulplay_choice_or_exit", side_effect=foulplay_choice),
            patch("pokezero.foulplay_bridge.action_index_from_choice_string", side_effect=decode),
        ):
            terminal = asyncio.run(
                _handle_decision_boundary(
                    config=config,
                    bridge=bridge,  # type: ignore[arg-type]
                    server=object(),
                    state=state,
                    policy=object(),
                    vocab=object(),
                    dex=object(),
                    observation_spec=SimpleNamespace(schema_version="v2.2"),
                    decision_round=0,
                    requested_players=("p1", "p2"),
                    foulplay_process=object(),
                    foulplay_logs=object(),
                )
            )

        self.assertIsNone(terminal)
        self.assertEqual(contexts[0].player_id, "p2")
        self.assertEqual(contexts[0].requested_legal_action_masks, {"p2": (True,) + (False,) * 8})
        self.assertEqual([step.player_id for step in state.trajectory.steps], ["p1", "p2"])
        self.assertEqual([decision.policy_id for decision in state.decisions], ["foul-play"])
        self.assertEqual(state.pokezero_decision_players, ["p2"])
        self.assertEqual(state.pokezero_submitted_choice_players, ["p2"])
        self.assertIn("policy_elapsed_seconds", state.decisions[0].metadata)
        self.assertEqual(state.decisions[0].metadata["policy_elapsed_seconds"], 2.5)
        self.assertEqual(bridge.messages[0]["choices"], {"p1": "move 1", "p2": "p2:0"})

    def test_benchmark_payload_records_terminal_and_policy_wall_telemetry(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=2,
        )
        result = ControlledFoulPlayBenchmarkResult(
            config=config,
            policy_id="checkpoint-raw",
            games=(
                ControlledFoulPlayGameResult(
                    battle_id="tie",
                    seed=1,
                    winner=None,
                    pokezero_won=False,
                    tied=True,
                    decision_rounds=2,
                    pokezero_decisions=2,
                    root_puct_searches=0,
                    root_puct_fallbacks=0,
                    policy_elapsed_seconds=(0.001, 0.003),
                ),
                ControlledFoulPlayGameResult(
                    battle_id="cap",
                    seed=2,
                    winner=None,
                    pokezero_won=False,
                    capped=True,
                    decision_rounds=250,
                    pokezero_decisions=2,
                    root_puct_searches=0,
                    root_puct_fallbacks=0,
                    policy_elapsed_seconds=(0.002, 0.006),
                ),
            ),
        )

        payload = result.to_dict()

        self.assertEqual(payload["ties"], 1)
        self.assertEqual(payload["capped_games"], 1)
        self.assertTrue(payload["game_results"][0]["tied"])
        self.assertFalse(payload["game_results"][0]["capped"])
        self.assertFalse(payload["game_results"][1]["tied"])
        self.assertTrue(payload["game_results"][1]["capped"])
        self.assertEqual(payload["game_results"][0]["pokezero_decision_players"], [])
        self.assertEqual(payload["game_results"][0]["pokezero_submitted_choice_players"], [])
        self.assertEqual(
            payload["policy_timing"],
            {
                "decision_count": 4,
                "total_elapsed_seconds": 0.012,
                "average_elapsed_seconds": 0.003,
                "p95_elapsed_seconds": 0.006,
            },
        )
        self.assertEqual(payload["score"], 1.0)
        self.assertEqual(payload["score_rate"], 0.5)
        self.assertEqual(
            payload["outcome_scoring"],
            {"win": 1.0, "tie": 0.5, "capped": 0.5, "loss": 0.0},
        )

    def test_benchmark_payload_preserves_root_puct_stage_timings(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=1,
            policy_mode="root-puct",
        )
        decision_timing = RootPUCTSearchTiming(
            opponent_scenario_planning_seconds=0.01,
            opponent_scenario_planning_count=1,
            observation_encoding_seconds=0.02,
            observation_encoding_count=2,
            neural_forward_seconds=0.03,
            neural_forward_count=2,
            total_seconds=0.06,
        ).to_dict()
        result = ControlledFoulPlayBenchmarkResult(
            config=config,
            policy_id="checkpoint-root-puct",
            games=(
                ControlledFoulPlayGameResult(
                    battle_id="timed",
                    seed=1,
                    winner="PokeZeroBot",
                    pokezero_won=True,
                    decision_rounds=2,
                    pokezero_decisions=1,
                    root_puct_searches=0,
                    root_puct_fallbacks=1,
                    root_puct_start_override_direct_materializations=3,
                    root_puct_start_override_replay_materializations=2,
                    root_puct_timings=(decision_timing,),
                ),
            ),
        )

        payload = result.to_dict()

        self.assertEqual(payload["game_results"][0]["root_puct_timing"], [decision_timing])
        self.assertEqual(
            payload["game_results"][0]["root_puct_start_override_direct_materializations"],
            3,
        )
        self.assertEqual(
            payload["game_results"][0]["root_puct_start_override_replay_materializations"],
            2,
        )
        self.assertEqual(payload["root_puct"]["timing"], decision_timing)

    def test_root_puct_timing_persistence_keeps_derived_timing_fields(self) -> None:
        persisted = _root_puct_timing_from_metadata(
            {
                "root_puct_timing": RootPUCTSearchTiming(
                    policy_evaluation_seconds=0.02,
                    policy_evaluation_count=1,
                    total_seconds=0.05,
                ).to_dict()
            }
        )

        self.assertIsNotNone(persisted)
        assert persisted is not None
        self.assertEqual(persisted["policy_value_evaluation_seconds"], 0.02)
        self.assertEqual(persisted["policy_value_evaluation_count"], 1)
        self.assertAlmostEqual(persisted["raw_residual_seconds"], 0.03)
        self.assertAlmostEqual(persisted["residual_seconds"], 0.03)

    def test_benchmark_payload_preserves_immutable_calibrated_leaf_provenance(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            value_checkpoint=Path("calibrated.pt"),
            showdown_root=Path("/showdown"),
        )
        provenance = {
            "policy_checkpoint_sha256": "raw-sha",
            "value_checkpoint_sha256": "leaf-sha",
            "value_calibration_source_checkpoint_sha256": "raw-sha",
            "value_calibration_transform": {"method": "isotonic"},
        }
        payload = ControlledFoulPlayBenchmarkResult(
            config=config,
            policy_id="checkpoint-root-puct",
            games=(),
            value_leaf_provenance=provenance,
        ).to_dict()

        self.assertEqual(payload["value_leaf"], provenance)

    def test_comparison_scores_ties_and_caps_as_half_points(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=2,
        )
        raw = ControlledFoulPlayBenchmarkResult(
            config=config,
            policy_id="raw",
            games=(
                ControlledFoulPlayGameResult(
                    battle_id="raw-tie",
                    seed=1,
                    winner=None,
                    pokezero_won=False,
                    tied=True,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=0,
                    root_puct_fallbacks=0,
                ),
                ControlledFoulPlayGameResult(
                    battle_id="raw-loss",
                    seed=2,
                    winner="FoulPlayBot",
                    pokezero_won=False,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=0,
                    root_puct_fallbacks=0,
                ),
            ),
        )
        root_puct = ControlledFoulPlayBenchmarkResult(
            config=ControlledFoulPlayConfig(
                checkpoint=config.checkpoint,
                showdown_root=config.showdown_root,
                games=2,
                policy_mode="root-puct",
            ),
            policy_id="root-puct",
            games=(
                ControlledFoulPlayGameResult(
                    battle_id="search-loss",
                    seed=1,
                    winner="FoulPlayBot",
                    pokezero_won=False,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=1,
                    root_puct_fallbacks=0,
                ),
                ControlledFoulPlayGameResult(
                    battle_id="search-cap",
                    seed=2,
                    winner=None,
                    pokezero_won=False,
                    capped=True,
                    decision_rounds=250,
                    pokezero_decisions=1,
                    root_puct_searches=1,
                    root_puct_fallbacks=0,
                ),
            ),
        )

        payload = ControlledFoulPlayComparisonResult(
            config=config,
            raw=raw,
            root_puct=root_puct,
        ).to_dict()

        aggregate = payload["comparison"]["aggregate"]["scored_outcomes"]
        paired = payload["comparison"]["paired_by_seed"]["scored_outcomes"]
        self.assertEqual(aggregate["raw"], {"games": 2, "score": 0.5, "score_rate": 0.25})
        self.assertEqual(aggregate["root_puct"], {"games": 2, "score": 0.5, "score_rate": 0.25})
        self.assertEqual(aggregate["root_puct_minus_raw_score_rate"], 0.0)
        self.assertEqual(paired["raw"], {"games": 2, "score": 0.5, "score_rate": 0.25})
        self.assertEqual(paired["root_puct"], {"games": 2, "score": 0.5, "score_rate": 0.25})
        self.assertEqual(paired["root_puct_minus_raw_score_rate"], 0.0)

    def test_final_allowed_choice_can_settle_to_a_terminal_result(self) -> None:
        class Bridge:
            def __init__(self) -> None:
                self.events = [
                    {
                        "type": "ready",
                        "battleId": "battle-gen3randombattle-controlled-7",
                        "requested": ["p1"],
                    },
                    {
                        "type": "terminal",
                        "battleId": "battle-gen3randombattle-controlled-7",
                    },
                ]

            async def send(self, _payload: dict) -> None:
                return None

            async def next_event(self) -> dict:
                return self.events.pop(0)

        class Server:
            async def send_room_lines(self, _battle_id: str, _lines: list[str]) -> None:
                return None

        async def handle_boundary(**_kwargs: object) -> None:
            return None

        async def notify_terminal(**_kwargs: object) -> None:
            return None

        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            max_decision_rounds=1,
        )
        with (
            patch("pokezero.foulplay_bridge._handle_decision_boundary", side_effect=handle_boundary),
            patch(
                "pokezero.foulplay_bridge._terminal_from_public_lines",
                return_value=TerminalState(winner="p1", turn_count=1),
            ),
            patch("pokezero.foulplay_bridge._notify_foulplay_terminal", side_effect=notify_terminal),
        ):
            result = asyncio.run(
                _run_single_game(
                    config=config,
                    bridge=Bridge(),  # type: ignore[arg-type]
                    server=Server(),  # type: ignore[arg-type]
                    policy=object(),
                    vocab=object(),
                    dex=object(),
                    observation_spec=object(),
                    seed=7,
                    foulplay_process=object(),
                    foulplay_logs=object(),
                )
            )

        self.assertEqual(result.winner, config.pokezero_username)
        self.assertTrue(result.pokezero_won)
        self.assertFalse(result.tied)
        self.assertFalse(result.capped)

    def test_public_corpus_rounds_use_protocol_identifiers_not_opponent_slots(self) -> None:
        state = _ControlledBattleState(
            battle_id="public-round",
            seed=7,
            format_id="gen3randombattle",
            public_lines=["|switch|p1a: Lead|Pikachu, L100|100/100"],
            trajectory=BattleTrajectory(
                battle_id="public-round",
                format_id="gen3randombattle",
                seed=7,
            ),
        )
        _capture_resolved_public_action_round(state, 0)
        state.previous_requested_players = ("p1", "p2")
        state.public_lines.extend(
            (
                "|move|p1a: Lead|Thunderbolt|p2a: Rival",
                "|move|p2a: Rival|Earthquake|p1a: Lead",
            )
        )

        _capture_resolved_public_action_round(state, 1)

        payload = state.public_resolved_action_rounds[0].to_dict()
        self.assertEqual(payload["actions"]["p1"], {"kind": "move", "move_id": "thunderbolt"})
        self.assertEqual(payload["actions"]["p2"], {"kind": "move", "move_id": "earthquake"})
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("action_index", serialized)
        self.assertNotIn("move_slot", serialized)
        self.assertNotIn("raw_choice", serialized)
        assert state.trajectory is not None
        self.assertEqual(state.trajectory.metadata["public_resolved_action_rounds"], [payload])

    def test_public_corpus_switch_identifier_uses_species_not_condition(self) -> None:
        state = _ControlledBattleState(
            battle_id="public-switch-round",
            seed=7,
            format_id="gen3randombattle",
        )
        state.previous_requested_players = ("p1",)
        state.public_lines.append("|switch|p1a: Lead|Pikachu, L100|243/243")

        _capture_resolved_public_action_round(state, 1)

        self.assertEqual(
            state.public_resolved_action_rounds[0].to_dict()["actions"]["p1"],
            {"kind": "switch", "switched_species": "pikachu"},
        )

    def test_public_corpus_drag_does_not_invent_a_voluntary_switch(self) -> None:
        state = _ControlledBattleState(
            battle_id="public-drag-round",
            seed=7,
            format_id="gen3randombattle",
        )
        state.previous_requested_players = ("p2",)
        state.public_lines.append("|drag|p2a: Rival|Pikachu, L100|243/243")

        _capture_resolved_public_action_round(state, 1)

        self.assertEqual(
            state.public_resolved_action_rounds[0].to_dict()["actions"]["p2"],
            {"kind": "event", "event_id": "unresolved-public-event"},
        )

    def test_capture_cli_requires_showdown_root(self) -> None:
        with self.assertRaises(SystemExit):
            asyncio.run(async_capture_main(["--checkpoint", "checkpoint.pt", "--out", "pool.jsonl"]))

    def test_player_state_can_request_turn_merged_transitions(self) -> None:
        state = _ControlledBattleState(
            battle_id="battle-gen3randombattle-controlled-1",
            seed=7,
            format_id="gen3randombattle",
        )
        expected = object()

        with (
            patch("pokezero.foulplay_bridge.parse_showdown_replay", return_value=object()),
            patch("pokezero.foulplay_bridge.normalize_for_player", return_value=expected) as normalize,
        ):
            actual = _player_state(state, "p1", set_source="source", include_turn_merged=True)

        self.assertIs(actual, expected)
        self.assertEqual(
            normalize.call_args.kwargs,
            {
                "player_id": "p1",
                "configured_showdown_slot": "p1",
                "format_id": "gen3randombattle",
                "set_source": "source",
                "include_turn_merged": True,
            },
        )

    def test_direct_materialization_state_excludes_the_opponent_request(self) -> None:
        state = _ControlledBattleState(
            battle_id="battle-gen3randombattle-controlled-1",
            seed=7,
            format_id="gen3randombattle",
            public_lines=[
                "|player|p1|PokeZero p1|",
                "|player|p2|FoulPlay|",
                "|switch|p1a: Charizard|Charizard, L80|250/250",
                "|switch|p2a: Xatu|Xatu, L80|220/220",
            ],
            request_lines={
                "p1": '|request|{"active":[{"moves":[]}],"side":{"id":"p1","pokemon":[]}}',
                "p2": '|request|{"active":[{"moves":[{"id":"psychic","pp":16}]}],"side":{"id":"p2"}}',
            },
            request_history=[
                ("p1", '|request|{"active":[{"moves":[]}],"side":{"id":"p1","pokemon":[]}}'),
                ("p2", '|request|{"active":[{"moves":[{"id":"psychic","pp":16}]}],"side":{"id":"p2"}}'),
            ],
        )

        materialization = _public_materialization_state(state, "p1")

        self.assertEqual(materialization.replay.requests, {})
        self.assertEqual(materialization.self_request["side"]["id"], "p1")
        self.assertEqual(materialization.self_move_states, {})
        self.assertEqual(materialization.self_initial_request["side"]["id"], "p1")
        self.assertNotIn("psychic", json.dumps(materialization.self_request))
        self.assertNotIn("psychic", json.dumps(materialization.self_move_states))
        self.assertNotIn("psychic", json.dumps(materialization.self_initial_request))
        self.assertEqual(materialization.replay.public_active["p2"].species, "Xatu")

    def test_the_self_move_states_populate_for_a_P2_seated_decider(self) -> None:
        """The BUILDER half of the p2 seat question, which engine_world tests cannot reach.

        `known_pp` in `engine_world._move_specs` is built from
        `payload["sides"][self]["pokemon"][i]["moves"]`, and those rows come from
        `self_move_states` here. A test that hand-writes those rows proves engine_world
        CONSUMES them symmetrically; only this one proves anything POPULATES them for p2.

        It matters because if `self_move_states` were empty for a p2 decider, `known_pp`
        would be empty, the self-moveset guard could not fire, and p2's own team would be
        built with full PP silently -- searching a state Showdown does not have instead of
        declining to search.

        Both seats are asserted from the same battle, so a builder that populated neither
        would fail rather than pass by symmetry.
        """
        def req(sid, species, move):
            return "|request|" + json.dumps({
                # pp != maxpp DELIBERATELY. With both at 16 the pp assertion below cannot
                # tell `"pp": pp` from `"pp": maxpp`, and that mutant reproduces the exact
                # silent-full-PP outcome this test exists to exclude.
                "active": [{"moves": [{"id": move, "pp": 16, "maxpp": 24}]}],
                "side": {"id": sid, "pokemon": [{
                    "ident": f"{sid}: {species}",
                    "details": f"{species}, L80",
                    "condition": "250/250",
                    "active": True,
                    "moves": [move],
                }]},
            })

        p1_request, p2_request = req("p1", "Charizard", "flamethrower"), req("p2", "Xatu", "psychic")
        state = _ControlledBattleState(
            battle_id="battle-gen3randombattle-controlled-1",
            seed=7,
            format_id="gen3randombattle",
            public_lines=[
                "|player|p1|PokeZero p1|",
                "|player|p2|FoulPlay|",
                "|switch|p1a: Charizard|Charizard, L80|250/250",
                "|switch|p2a: Xatu|Xatu, L80|220/220",
            ],
            request_lines={"p1": p1_request, "p2": p2_request},
            request_history=[("p1", p1_request), ("p2", p2_request)],
        )

        for seat, expected_move in (("p1", "flamethrower"), ("p2", "psychic")):
            materialization = _public_materialization_state(state, seat)
            self.assertTrue(
                materialization.self_move_states,
                f"self_move_states is EMPTY for a {seat}-seated decider, so known_pp "
                f"would be empty and the self-moveset guard could not fire: "
                f"{materialization.self_move_states!r}",
            )
            retained = [
                move["id"]
                for moves in materialization.self_move_states.values()
                for move in moves
            ]
            self.assertIn(expected_move, retained)
            # ...and the PP is the REMAINING pp (16), not the maximum (24), which is what
            # `known_pp` reads and what distinguishes a real request read from full PP.
            pps = {
                move["id"]: move.get("pp")
                for moves in materialization.self_move_states.values()
                for move in moves
            }
            self.assertEqual(pps[expected_move], 16)
            # The OPPONENT's move must never appear on the self side.
            other = "psychic" if seat == "p1" else "flamethrower"
            self.assertNotIn(other, retained, f"{seat} self side leaked the opponent's move")

    def test_capture_writes_p1_only_rollouts_and_preserves_partial_output(self) -> None:
        # NAMES v2.2 so the observation's SHAPE and its STAMP come from the same schema. This test
        # builds a PokeZeroObservationV0 with widths taken from `spec` and lets `schema_version` fall
        # to the dataclass default, which NAMES v2.2 since #1244. While the process default was also
        # v2.2 those agreed; under a v4 rotation the widths came from v4 and the stamp stayed v2.2, and
        # the assertion below -- `observation.schema_version == spec.schema_version`, deliberately
        # relative rather than hardcoded -- failed `'v2.2' != 'v4'`. The drill verdict recorded this as
        # "should be rotation-invariant, and is not; something in the path binds a fixed spec while the
        # capture uses the default". That something is the field default, and naming the spec here
        # closes the gap without weakening the relative comparison.
        spec = V2_2_REPLAY_OBSERVATION_SPEC
        observation = PokeZeroObservationV0(
            categorical_ids=tuple(
                tuple(0 for _ in range(spec.categorical_feature_count))
                for _ in range(spec.token_count)
            ),
            numeric_features=tuple(
                tuple(0.0 for _ in range(spec.numeric_feature_count))
                for _ in range(spec.token_count)
            ),
            token_type_ids=tuple(0 for _ in range(spec.token_count)),
            attention_mask=tuple(True for _ in range(spec.token_count)),
            legal_action_mask=tuple(index == 0 for index in range(ACTION_COUNT)),
            metadata={
                "belief_view": {
                    "self_slot": "p1",
                    "opponent_slot": "p2",
                    "self_pokemon": [],
                    "opponent_pokemon": [],
                }
            },
        )
        trajectory = BattleTrajectory(battle_id="capture-1", format_id="gen3randombattle", seed=17)
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=observation,
                legal_action_mask=observation.legal_action_mask,
                action_index=0,
                metadata={},
            )
        )
        trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=observation,
                legal_action_mask=observation.legal_action_mask,
                action_index=0,
                metadata={},
            )
        )
        trajectory.record_terminal(TerminalState(winner="p1", turn_count=1, capped=False))
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "checkpoint.pt"
            checkpoint.write_bytes(b"capture-checkpoint")
            config = ControlledFoulPlayConfig(
                checkpoint=checkpoint,
                showdown_root=Path("/showdown"),
                policy_mode="raw",
                belief_set_source=False,
            )

            async def fake_benchmark(*_args, **kwargs):
                kwargs["trajectory_callback"](trajectory)
                return ControlledFoulPlayBenchmarkResult(config=config, policy_id="raw", games=())

            out_path = Path(tmp_dir) / "pool.jsonl"
            public_corpus_path = Path(tmp_dir) / "public.jsonl"
            with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", side_effect=fake_benchmark):
                result = asyncio.run(
                    capture_controlled_foulplay_rollouts(
                        config,
                        out_path=out_path,
                        pool_id="step0",
                        public_corpus_out=public_corpus_path,
                    )
                )

            records = list(read_rollout_records(out_path))
            self.assertEqual(len(records), 1)
            self.assertEqual([step.player_id for step in records[0].trajectory.steps], ["p1"])
            self.assertEqual(records[0].trajectory.metadata["capture"], "controlled-foulplay/raw")
            self.assertEqual(records[0].trajectory.metadata["pool"], "step0")
            self.assertEqual(records[0].trajectory.steps[0].observation.schema_version, spec.schema_version)
            self.assertEqual(len(records[0].trajectory.steps[0].observation.numeric_features[0]), spec.numeric_feature_count)
            self.assertEqual(result.captured_games, 1)
            self.assertEqual(result.skipped_capped_games, 0)
            self.assertTrue(result.checkpoint_sha256)
            self.assertEqual(result.captured_public_decisions, 1)
            public_corpus = load_public_decision_corpus(public_corpus_path)
            self.assertEqual(len(public_corpus.decisions), 1)
            self.assertEqual(public_corpus.decisions[0].acting_player, "p1")
            self.assertEqual(public_corpus.manifest["opponent_legal_mask_mode"], "hidden")

            if torch_available():
                torch = require_torch()

                class FixedValueModel:
                    def eval(self) -> None:
                        pass

                    def __call__(self, **kwargs):
                        batch_size = int(kwargs["categorical_ids"].shape[0])
                        return SimpleNamespace(value=torch.full((batch_size,), 0.25))

                report = evaluate_value_calibration(
                    model=FixedValueModel(),
                    training_result=SimpleNamespace(training_config=TransformerTrainingConfig(window_size=1)),
                    paths=out_path,
                    batch_size=1,
                    bins=2,
                )
                self.assertEqual(report.examples, 1)
                self.assertEqual(report.sign_accuracy, 1.0)

            with self.assertRaises(FileExistsError):
                asyncio.run(capture_controlled_foulplay_rollouts(config, out_path=out_path, pool_id="step0"))

    def test_capture_does_not_create_an_output_file_before_the_first_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "checkpoint.pt"
            checkpoint.write_bytes(b"capture-checkpoint")
            config = ControlledFoulPlayConfig(
                checkpoint=checkpoint,
                showdown_root=Path("/showdown"),
                policy_mode="raw",
                belief_set_source=False,
            )
            out_path = Path(tmp_dir) / "pool.jsonl"

            async def failing_benchmark(*_args, **_kwargs):
                raise RuntimeError("foul-play failed before the first completed game")

            with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", side_effect=failing_benchmark):
                with self.assertRaisesRegex(RuntimeError, "before the first"):
                    asyncio.run(capture_controlled_foulplay_rollouts(config, out_path=out_path))

            self.assertFalse(out_path.exists())

    def test_capture_excludes_capped_games_from_value_labels(self) -> None:
        observation = PokeZeroObservationV0(
            categorical_ids=(),
            numeric_features=(),
            token_type_ids=(),
            attention_mask=(),
            legal_action_mask=tuple(index == 0 for index in range(ACTION_COUNT)),
        )
        trajectory = BattleTrajectory(battle_id="capped-1", format_id="gen3randombattle", seed=19)
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=observation,
                legal_action_mask=observation.legal_action_mask,
                action_index=0,
                metadata={},
            )
        )
        trajectory.record_terminal(TerminalState(winner=None, turn_count=250, capped=True))

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "checkpoint.pt"
            checkpoint.write_bytes(b"capture-checkpoint")
            config = ControlledFoulPlayConfig(
                checkpoint=checkpoint,
                showdown_root=Path("/showdown"),
                policy_mode="raw",
                belief_set_source=False,
            )
            out_path = Path(tmp_dir) / "pool.jsonl"

            async def fake_benchmark(*_args, **kwargs):
                kwargs["trajectory_callback"](trajectory)
                return ControlledFoulPlayBenchmarkResult(config=config, policy_id="raw", games=())

            with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", side_effect=fake_benchmark):
                result = asyncio.run(capture_controlled_foulplay_rollouts(config, out_path=out_path))

            self.assertEqual(result.captured_games, 0)
            self.assertEqual(result.skipped_capped_games, 1)
            self.assertFalse(out_path.exists())

    def test_capture_rejects_search_policy_mode(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            policy_mode="root-puct",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "policy_mode='raw'"):
                asyncio.run(
                    capture_controlled_foulplay_rollouts(config, out_path=Path(tmp_dir) / "pool.jsonl")
                )

    def test_capture_rejects_the_p2_mirrored_seat(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            policy_mode="raw",
            pokezero_player="p2",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "pokezero_player='p1'"):
                asyncio.run(
                    capture_controlled_foulplay_rollouts(config, out_path=Path(tmp_dir) / "pool.jsonl")
                )

    def test_split_outgoing_showdown_message_handles_room_and_global(self) -> None:
        self.assertEqual(
            _split_outgoing_showdown_message("battle-gen3randombattle-1|/choose move surf|7"),
            ("battle-gen3randombattle-1", "/choose move surf|7"),
        )
        self.assertEqual(
            _split_outgoing_showdown_message("|/trn FoulPlayBot,0,"),
            ("", "/trn FoulPlayBot,0,"),
        )

    def test_choice_body_from_foulplay_messages_normalizes_move_and_switch(self) -> None:
        self.assertEqual(_choice_body_from_outgoing_message("/choose move thunderbolt|12"), "move thunderbolt")
        self.assertEqual(_choice_body_from_outgoing_message("/choose move 3|12"), "move 3")
        self.assertEqual(_choice_body_from_outgoing_message("/switch 4|12"), "switch 4")
        self.assertIsNone(_choice_body_from_outgoing_message("/timer on"))

    def test_line_chunks_safe_for_foulplay_drops_noise_and_splits_sensitive_lines(self) -> None:
        self.assertEqual(
            _line_chunks_safe_for_foulplay(
                (
                    "|t:|1783052150",
                    "|",
                    "|gametype|singles",
                    "|player|p1|PokeZeroBot|",
                    "|start",
                    "|request|{}",
                )
            ),
            (
                ("|gametype|singles",),
                ("|player|p1|PokeZeroBot|",),
                ("|start",),
                ("|request|{}",),
            ),
        )

    def test_line_for_foulplay_injects_rqid_into_battlestream_request_copy(self) -> None:
        state = _ControlledBattleState(
            battle_id="battle-gen3randombattle-1",
            seed=1,
            format_id="gen3randombattle",
        )

        line = _line_for_foulplay(state, '|request|{"active":[{"moves":[]}],"side":{"id":"p2"}}')

        self.assertIn('"rqid":1', line)
        self.assertEqual(state.next_foulplay_rqid, 2)
        self.assertEqual(_line_for_foulplay(state, '|request|{"rqid":99}'), '|request|{"rqid":99}')

    def test_requested_legal_action_masks_can_hide_opponent_private_mask(self) -> None:
        class Observation:
            def __init__(self, mask: tuple[bool, ...]) -> None:
                self.legal_action_mask = mask

        observations = {
            "p1": Observation((True, False)),
            "p2": Observation((False, True)),
        }

        self.assertEqual(
            _requested_legal_action_masks_for_context(
                observations,  # type: ignore[arg-type]
                acting_player="p1",
                opponent_legal_mask_mode="hidden",
            ),
            {"p1": (True, False)},
        )
        self.assertEqual(
            _requested_legal_action_masks_for_context(
                observations,  # type: ignore[arg-type]
                acting_player="p1",
                opponent_legal_mask_mode="privileged",
            ),
            {"p1": (True, False), "p2": (False, True)},
        )

    def test_terminal_line_for_foulplay_uses_configured_display_names(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
        )

        self.assertEqual(
            _terminal_line_for_foulplay(TerminalState(winner="p1", turn_count=10), config),
            "|win|PokeZeroBot",
        )
        self.assertEqual(
            _terminal_line_for_foulplay(TerminalState(winner="p2", turn_count=10), config),
            "|win|FoulPlayBot",
        )
        self.assertEqual(
            _terminal_line_for_foulplay(TerminalState(winner=None, turn_count=250, capped=True), config),
            "|tie|",
        )
        mirrored = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            pokezero_player="p2",
        )
        self.assertEqual(
            _terminal_line_for_foulplay(TerminalState(winner="p1", turn_count=10), mirrored),
            "|win|FoulPlayBot",
        )
        self.assertEqual(
            _terminal_line_for_foulplay(TerminalState(winner="p2", turn_count=10), mirrored),
            "|win|PokeZeroBot",
        )

    def test_is_terminal_protocol_line_detects_win_and_tie(self) -> None:
        self.assertTrue(_is_terminal_protocol_line("|win|PokeZeroBot"))
        self.assertTrue(_is_terminal_protocol_line("|tie|"))
        self.assertFalse(_is_terminal_protocol_line("|turn|2"))

    def test_belief_set_source_gate_honors_env_and_explicit_override(self) -> None:
        # Regression: benchmarks silently evaluated nets with candidate-set features ablated while
        # training ran with them enabled (train/eval observation mismatch). The gate must default
        # to the shared POKEZERO_BELIEF_SET_SOURCE env flip point and allow explicit override.
        import os

        config = ControlledFoulPlayConfig(checkpoint=Path("checkpoint.pt"), showdown_root=Path("/showdown"))
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("POKEZERO_BELIEF_SET_SOURCE", None)
            self.assertFalse(config.belief_set_source_enabled())
        with patch.dict(os.environ, {"POKEZERO_BELIEF_SET_SOURCE": "1"}):
            self.assertTrue(config.belief_set_source_enabled())
            forced_off = ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                belief_set_source=False,
            )
            self.assertFalse(forced_off.belief_set_source_enabled())
        forced_on = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            belief_set_source=True,
        )
        self.assertTrue(forced_on.belief_set_source_enabled())

    def test_provenance_mismatch_warning_three_way_and_dedup(self) -> None:
        import contextlib
        import io

        from pokezero.foulplay_bridge import _PROVENANCE_WARNINGS_EMITTED, _warn_on_belief_provenance_mismatch

        class Recorded:
            def __init__(self, value):
                self.belief_set_source_hash = value

        class FakeSource:
            class metadata:
                source_hash = "currenthash0"

        config_off = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"), showdown_root=Path("/showdown"), belief_set_source=False
        )
        config_on = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"), showdown_root=Path("/showdown"), belief_set_source=True
        )

        def warn_output(config, result) -> str:
            _PROVENANCE_WARNINGS_EMITTED.clear()
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                _warn_on_belief_provenance_mismatch(config, result)
            return stderr.getvalue()

        # matched (both off) -> silent; result lacking the attribute behaves as None
        self.assertEqual(warn_output(config_off, Recorded(None)), "")
        self.assertEqual(warn_output(config_off, object()), "")
        # recorded but benchmark disabled
        self.assertIn("runs with it disabled", warn_output(config_off, Recorded("trainedhash1")))
        with patch("pokezero.foulplay_bridge._resolved_belief_set_source", return_value=FakeSource()):
            # legacy checkpoint, benchmark enabled -> message names the enabled side
            out = warn_output(config_on, Recorded(None))
            self.assertIn("no belief provenance", out)
            self.assertIn("enabled", out)
            # both set, different hashes
            self.assertIn("!=", warn_output(config_on, Recorded("trainedhash1")))
            # matched hashes -> silent
            self.assertEqual(warn_output(config_on, Recorded("currenthash0")), "")
            # dedup: identical (checkpoint, condition) warns once per process
            _PROVENANCE_WARNINGS_EMITTED.clear()
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                _warn_on_belief_provenance_mismatch(config_on, Recorded("trainedhash1"))
                _warn_on_belief_provenance_mismatch(config_on, Recorded("trainedhash1"))
            self.assertEqual(stderr.getvalue().count("warning:"), 1)

    def test_config_rejects_invalid_search_tuning_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection_mode"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                selection_mode="unknown",
            )
        with self.assertRaisesRegex(ValueError, "minimum_value_improvement"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                minimum_value_improvement=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "minimum_override_prior_ratio"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                minimum_override_prior_ratio=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "minimum_override_prior_ratio"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                minimum_override_prior_ratio=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "minimum_score_improvement"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                minimum_score_improvement=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "root_visit_budget"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                root_visit_budget=0,
            )
        with self.assertRaisesRegex(ValueError, "root_prior_temperature"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                root_prior_temperature=0.0,
            )
        with self.assertRaisesRegex(ValueError, "root_time_budget_ms"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                root_time_budget_ms=0,
            )
        with self.assertRaisesRegex(ValueError, "root_opponent_action_candidate_scenarios"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                root_opponent_action_candidate_scenarios=0,
            )
        with self.assertRaisesRegex(ValueError, "root_opponent_action_candidate_scenarios"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                root_opponent_action_scenarios=2,
                root_opponent_action_candidate_scenarios=1,
            )
        with self.assertRaisesRegex(ValueError, "leaf_rollout_sampling"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                leaf_rollout_sampling=True,
            )
        with self.assertRaisesRegex(ValueError, "start_override_attempts"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                start_override_attempts=0,
            )
        with self.assertRaisesRegex(ValueError, "belief_start_override_samples"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                belief_start_override_samples=0,
            )
        with self.assertRaisesRegex(ValueError, "belief_start_override_samples"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                belief_start_override_samples=2,
            )
        with self.assertRaisesRegex(ValueError, "foulplay_random_seed"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                foulplay_random_seed=-1,
            )

    def test_controlled_foulplay_defaults_to_visit_selection(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
        )
        args = build_arg_parser().parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--showdown-root",
                "/showdown",
            ]
        )

        self.assertEqual(config.selection_mode, "visits")
        self.assertEqual(config.root_visit_budget, 16)
        self.assertIsNone(config.root_extra_visits)
        self.assertIsNone(config.value_checkpoint)
        self.assertIsNone(config.root_prior_temperature)
        self.assertEqual(config.effective_root_prior_temperature, 1.0)
        self.assertEqual(config.root_opponent_action_scenarios, 1)
        self.assertEqual(config.root_opponent_action_candidate_scenarios, ACTION_COUNT)
        self.assertEqual(config.start_override_attempts, 10)
        self.assertEqual(config.belief_start_override_samples, 1)
        self.assertEqual(args.selection_mode, "visits")
        self.assertEqual(args.root_visit_budget, 16)
        self.assertIsNone(args.root_extra_visits)
        self.assertIsNone(args.value_checkpoint)
        self.assertIsNone(args.root_prior_temperature)
        self.assertEqual(args.root_opponent_action_scenarios, 1)
        self.assertEqual(args.root_opponent_action_candidate_scenarios, ACTION_COUNT)
        self.assertEqual(args.start_override_attempts, 10)
        self.assertEqual(args.belief_start_override_samples, 1)
        sampled_args = build_arg_parser().parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--showdown-root",
                "/showdown",
                "--belief-start-overrides",
                "--belief-start-override-samples",
                "3",
            ]
        )
        self.assertTrue(sampled_args.belief_start_overrides)
        self.assertEqual(sampled_args.belief_start_override_samples, 3)

        warmed_config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            temperature=1.75,
        )
        self.assertEqual(warmed_config.effective_root_prior_temperature, 1.75)

    def test_controlled_foulplay_supports_fixed_extra_and_adaptive_root_budgets(self) -> None:
        fixed = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            value_checkpoint=Path("calibrated.pt"),
            showdown_root=Path("/showdown"),
            root_extra_visits=24,
        )
        self.assertEqual(
            fixed.root_visit_budget_selector().to_dict(),
            {"selector_id": "fixed-extra-visits", "extra_visits": 24},
        )

        adaptive = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            adaptive_root_contested_extra_visits=120,
            adaptive_root_policy_entropy_threshold=0.7,
        )
        self.assertEqual(
            adaptive.root_visit_budget_selector().to_dict(),
            {
                "selector_id": "entropy-or-value-margin",
                "contested_extra_visits": 120,
                "uncontested_extra_visits": 0,
                "minimum_policy_entropy": 0.7,
                "maximum_value_margin": None,
            },
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                root_extra_visits=24,
                adaptive_root_contested_extra_visits=120,
                adaptive_root_policy_entropy_threshold=0.7,
            )
        with self.assertRaisesRegex(ValueError, "root_time_budget_ms cannot"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                root_extra_visits=24,
                root_time_budget_ms=100,
            )
        time_bounded = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            root_time_budget_ms=100,
        )
        self.assertIsNone(time_bounded.root_visit_budget)
        parsed_time_bounded = _config_from_args(
            build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    "checkpoint.pt",
                    "--showdown-root",
                    "/showdown",
                    "--root-visit-budget",
                    "32",
                    "--root-time-budget-ms",
                    "100",
                ]
            )
        )
        self.assertIsNone(parsed_time_bounded.root_visit_budget)

    def test_build_policy_uses_full_action_default_opponent_candidate_reserve(self) -> None:
        class FakePolicy:
            def __init__(self, policy_id: str | None = None, **_: object) -> None:
                self.policy_id = policy_id or "fake-transformer"

        fake_result = type(
            "FakeTrainingResult",
            (),
            {"model_config": type("FakeModelConfig", (), {"policy_id": "fake-base"})()},
        )()
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
        )

        with patch("pokezero.foulplay_bridge.TransformerSoftmaxPolicy", side_effect=FakePolicy):
            policy = _build_policy(
                config=config,
                model=object(),
                result=fake_result,
                value_model=object(),
                value_result=fake_result,
                env_config=object(),
                rollout_config=object(),
                policy_id="fake-base",
            )

        self.assertEqual(
            getattr(policy.opponent_action_scenario_planner, "planner_id"),
            f"checkpoint-top{ACTION_COUNT}",
        )
        self.assertEqual(policy.max_opponent_action_scenarios, 1)
        self.assertEqual(policy.start_override_samples_per_scenario, 1)

    def test_build_policy_binds_raw_checkpoint_provenance_not_value_leaf(self) -> None:
        class FakePolicy:
            def __init__(
                self,
                policy_id: str | None = None,
                checkpoint_path: str | None = None,
                weights_sha256: str | None = None,
                **_: object,
            ) -> None:
                self.policy_id = policy_id or "fake-transformer"
                self.checkpoint_path = checkpoint_path
                self.weights_sha256 = weights_sha256

        fake_result = type(
            "FakeTrainingResult",
            (),
            {"model_config": type("FakeModelConfig", (), {"policy_id": "fake-base"})()},
        )()
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_checkpoint = Path(temp_dir) / "raw.pt"
            raw_checkpoint.write_bytes(b"raw-policy")
            config = ControlledFoulPlayConfig(
                checkpoint=raw_checkpoint,
                value_checkpoint=Path(temp_dir) / "distinct-value-leaf.pt",
                showdown_root=Path("/showdown"),
            )
            with patch("pokezero.foulplay_bridge.TransformerSoftmaxPolicy", side_effect=FakePolicy):
                policy = _build_policy(
                    config=config,
                    model=object(),
                    result=fake_result,
                    value_model=object(),
                    value_result=fake_result,
                    env_config=object(),
                    rollout_config=object(),
                    policy_id="fake-base",
                )

        self.assertEqual(policy.checkpoint_path, str(raw_checkpoint.resolve()))
        self.assertEqual(policy.weights_sha256, hashlib.sha256(b"raw-policy").hexdigest())
        self.assertEqual(policy.fallback_policy.checkpoint_path, str(raw_checkpoint.resolve()))
        self.assertEqual(policy.fallback_policy.weights_sha256, policy.weights_sha256)

    def test_build_policy_uses_separate_calibrated_value_model_and_relative_budget(self) -> None:
        class FakePolicy:
            def __init__(self, policy_id: str | None = None, **_: object) -> None:
                self.policy_id = policy_id or "fake-transformer"

        fake_result = type(
            "FakeTrainingResult",
            (),
            {"model_config": type("FakeModelConfig", (), {"policy_id": "fake-base"})()},
        )()
        policy_model = object()
        value_model = object()
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            value_checkpoint=Path("calibrated.pt"),
            showdown_root=Path("/showdown"),
            root_extra_visits=24,
        )

        with (
            patch("pokezero.foulplay_bridge.TransformerSoftmaxPolicy", side_effect=FakePolicy),
            patch("pokezero.foulplay_bridge.evaluate_transformer_observation_value", return_value=0.5) as value_eval,
        ):
            policy = _build_policy(
                config=config,
                model=policy_model,
                result=fake_result,
                value_model=value_model,
                value_result=fake_result,
                env_config=object(),
                rollout_config=object(),
                policy_id="fake-base",
            )
            self.assertEqual(
                policy.value_fn(
                    (
                        PokeZeroObservationV0(
                            categorical_ids=(),
                            numeric_features=(),
                            token_type_ids=(),
                            attention_mask=(),
                            legal_action_mask=(),
                        ),
                    )
                ),
                0.5,
            )

        self.assertIs(value_eval.call_args.kwargs["model"], value_model)
        self.assertIsNotNone(value_eval.call_args.kwargs["timing"])
        self.assertIsNotNone(policy.neural_timing_snapshot)
        self.assertEqual(
            policy.root_visit_budget_selector.to_dict(),
            {"selector_id": "fixed-extra-visits", "extra_visits": 24},
        )

    def test_build_policy_wires_belief_start_override_samples(self) -> None:
        class FakePolicy:
            def __init__(self, policy_id: str | None = None, **_: object) -> None:
                self.policy_id = policy_id or "fake-transformer"

        fake_result = type(
            "FakeTrainingResult",
            (),
            {"model_config": type("FakeModelConfig", (), {"policy_id": "fake-base"})()},
        )()
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            belief_start_overrides=True,
            belief_start_override_samples=3,
        )

        with patch("pokezero.foulplay_bridge.TransformerSoftmaxPolicy", side_effect=FakePolicy), patch(
            "pokezero.foulplay_bridge.load_gen3_randbat_source_cached",
            return_value=object(),
        ), patch(
            "pokezero.foulplay_bridge.gen3_randbat_belief_start_override_planner",
            return_value=lambda context, scenario, scenario_index, rng: None,
        ):
            policy = _build_policy(
                config=config,
                model=object(),
                result=fake_result,
                value_model=object(),
                value_result=fake_result,
                env_config=object(),
                rollout_config=object(),
                policy_id="fake-base",
            )

        self.assertEqual(policy.start_override_samples_per_scenario, 3)
        self.assertEqual(policy.max_opponent_action_scenarios, 1)

    def test_build_policy_hands_the_selection_knobs_to_the_search_config(self) -> None:
        """The last bridge-side link of the selection-tuning chain.

        `engine_c_puct` and `engine_fpu_reduction` are inert unless this
        construction copies them onto `EngineMctsConfig`, which is the object
        `native_search_args` reads (c_puct at positional slot 8, fpu behind the
        widening cascade). A dropped assignment here does not fail: the shard
        completes, at the default, under a config_id that claims otherwise.
        """
        import pokezero.engine_search as engine_search

        captured = {}

        class FakeEnginePolicy:
            def __init__(self, *, config=None, policy_id=None, **_: object) -> None:
                captured["config"] = config
                self.policy_id = policy_id

        fake_result = type(
            "FakeTrainingResult",
            (),
            {"model_config": type("FakeModelConfig", (), {"policy_id": "fake-base"})()},
        )()
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            policy_mode="engine-mcts",
            engine_model_path=Path("/art/model_ts.pt"),
            engine_tables_path=Path("/art/encoder_tables.json"),
            engine_c_puct=0.8,
            engine_fpu_reduction=0.2,
        )

        with patch.object(engine_search, "EngineMctsPolicy", FakeEnginePolicy), patch(
            "pokezero.foulplay_bridge.load_showdown_dex_cached", return_value=object()
        ), patch(
            "pokezero.foulplay_bridge.load_gen3_randbat_source_cached",
            return_value=object(),
        ):
            _build_policy(
                config=config,
                model=object(),
                result=fake_result,
                value_model=object(),
                value_result=fake_result,
                env_config=object(),
                rollout_config=object(),
                policy_id="fake-base",
            )

        self.assertEqual(captured["config"].c_puct, 0.8)
        self.assertEqual(captured["config"].fpu_reduction, 0.2)

    def test_build_policy_leaves_the_selection_knobs_at_their_recorded_defaults(self) -> None:
        # The other half: an untuned cell must reach the crate as the search
        # every banked result was produced under -- flat 0.5 FPU, c_puct 1.4.
        import pokezero.engine_search as engine_search

        captured = {}

        class FakeEnginePolicy:
            def __init__(self, *, config=None, **_: object) -> None:
                captured["config"] = config

        fake_result = type(
            "FakeTrainingResult",
            (),
            {"model_config": type("FakeModelConfig", (), {"policy_id": "fake-base"})()},
        )()
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            policy_mode="engine-mcts",
            engine_model_path=Path("/art/model_ts.pt"),
            engine_tables_path=Path("/art/encoder_tables.json"),
        )

        with patch.object(engine_search, "EngineMctsPolicy", FakeEnginePolicy), patch(
            "pokezero.foulplay_bridge.load_showdown_dex_cached", return_value=object()
        ), patch(
            "pokezero.foulplay_bridge.load_gen3_randbat_source_cached",
            return_value=object(),
        ):
            _build_policy(
                config=config,
                model=object(),
                result=fake_result,
                value_model=object(),
                value_result=fake_result,
                env_config=object(),
                rollout_config=object(),
                policy_id="fake-base",
            )

        self.assertEqual(captured["config"].c_puct, 1.4)
        self.assertIsNone(captured["config"].fpu_reduction)

    def test_foulplay_process_command_seeds_python_random(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            foulplay_root=Path("/foul-play"),
            foulplay_python=Path("/python"),
            seed_start=123,
            foulplay_random_seed=456,
            games=7,
            search_time_ms=10,
        )

        command = _foulplay_command(config, "ws://127.0.0.1:1/showdown/websocket")
        env = _foulplay_env(config)

        self.assertEqual(command[0], "/python")
        self.assertEqual(command[1], "-c")
        self.assertIn("random.seed", command[2])
        self.assertIn("runpy.run_path", command[2])
        self.assertIn("/foul-play/run.py", command)
        self.assertIn("--run-count", command)
        self.assertEqual(command[command.index("--run-count") + 1], "7")
        self.assertEqual(env["POKEZERO_FOULPLAY_RANDOM_SEED"], "456")
        self.assertEqual(env["PYTHONHASHSEED"], "456")
        self.assertEqual(env["FOULPLAY_LOCAL_NOSEC"], "1")

    def test_foulplay_process_seed_wrapper_executes_target_with_expected_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            foulplay_root = Path(temp_dir)
            run_py = foulplay_root / "run.py"
            run_py.write_text(
                "import json, random, sys\n"
                "print(json.dumps({'argv': sys.argv, 'draw': random.random()}))\n"
            )
            config = ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                foulplay_root=foulplay_root,
                foulplay_python=Path(sys.executable),
                seed_start=123,
                foulplay_random_seed=456,
                games=7,
                search_time_ms=10,
            )

            completed = subprocess.run(
                _foulplay_command(config, "ws://127.0.0.1:1/showdown/websocket"),
                cwd=foulplay_root,
                env=_foulplay_env(config),
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            payload = json.loads(completed.stdout)

        self.assertEqual(payload["argv"][0], str(run_py))
        self.assertIn("--websocket-uri", payload["argv"])
        self.assertEqual(payload["argv"][payload["argv"].index("--run-count") + 1], "7")
        self.assertEqual(payload["draw"], random.Random(456).random())

    def test_foulplay_process_seed_defaults_to_seed_start(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            seed_start=321,
        )

        self.assertEqual(config.resolved_foulplay_random_seed, 321)
        self.assertEqual(_foulplay_env(config)["POKEZERO_FOULPLAY_RANDOM_SEED"], "321")
        self.assertEqual(_foulplay_env(config)["PYTHONHASHSEED"], "321")

    def test_foulplay_hash_seed_is_clamped_to_python_supported_range(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            foulplay_random_seed=(2**32) + 5,
        )

        self.assertEqual(_foulplay_env(config)["POKEZERO_FOULPLAY_RANDOM_SEED"], str((2**32) + 5))
        self.assertEqual(_foulplay_env(config)["PYTHONHASHSEED"], "5")

    def test_benchmark_payload_summarizes_root_puct_metrics(self) -> None:
        mixed_replay_reason = (
            "all opponent action scenarios were replay-illegal: "
            "replay actions for decision round 12 do not match environment request "
            "(unexpected players: p2); "
            "start override does not reproduce recorded replay prefix observations "
            "for decision round 28: p1. "
            "(numeric_features/opponent_pokemon[8][0]: actual=0.75 expected=1.0)"
        )
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=2,
            selection_mode="visits",
            minimum_value_improvement=0.25,
            minimum_override_prior_ratio=0.5,
            minimum_score_improvement=0.1,
            root_prior_temperature=2.5,
            root_visit_budget=16,
            root_time_budget_ms=250,
            root_opponent_action_scenarios=2,
            root_opponent_action_candidate_scenarios=5,
            leaf_rollout_rounds=1,
            leaf_rollout_sampling=True,
            belief_start_overrides=True,
            start_override_attempts=7,
            belief_start_override_samples=3,
        )
        result = ControlledFoulPlayBenchmarkResult(
            config=config,
            policy_id="policy+root-puct",
            games=(
                ControlledFoulPlayGameResult(
                    battle_id="battle-1",
                    seed=1,
                    winner="PokeZeroBot",
                    pokezero_won=True,
                    decision_rounds=3,
                    pokezero_decisions=3,
                    root_puct_searches=3,
                    root_puct_fallbacks=0,
                    root_puct_total_visits=24,
                    root_puct_effective_total_visits=18,
                    root_puct_opponent_action_scenarios_generated=9,
                    root_puct_opponent_action_scenarios_skipped=1,
                    root_puct_opponent_action_scenarios_unsearched=2,
                    root_puct_opponent_action_skip_categories={
                        "start_override_observation_mismatch": 1,
                    },
                    root_puct_opponent_action_missing_sampled_world_reason_categories={
                        "self_team_unavailable": 2,
                    },
                    root_puct_opponent_action_replay_rejection_decision_rounds={"3": 1},
                    root_puct_opponent_action_start_override_mismatch_decision_rounds={"3": 1},
                    root_puct_opponent_action_first_observation_mismatch_paths={
                        "categorical_ids/opponent_pokemon[8][11]": 1,
                    },
                    root_puct_opponent_action_groups_generated=5,
                    root_puct_opponent_action_groups_used=3,
                    root_puct_opponent_action_groups_skipped=1,
                    root_puct_opponent_action_groups_unsearched=1,
                    root_puct_selected_prior_action_changes=2,
                    root_puct_pre_gate_prior_action_changes=3,
                    root_puct_time_budget_exhaustions=2,
                    root_puct_start_override_sources_used=3,
                    root_puct_start_override_attempts_used=5,
                    root_puct_start_override_duplicate_attempts=1,
                    root_puct_start_override_shared_samples=6,
                    root_puct_start_override_shared_samples_accepted=4,
                    root_puct_start_override_shared_samples_rejected=2,
                    root_puct_prior_action_change_details=(
                        {
                            "decision_index": 1,
                            "selected_action": 4,
                            "search_action": 4,
                            "prior_action": 0,
                            "selected_changed_prior_action": True,
                            "pre_gate_changed_prior_action": True,
                        },
                    ),
                    root_puct_average_elapsed_seconds=0.2,
                ),
                ControlledFoulPlayGameResult(
                    battle_id="battle-2",
                    seed=2,
                    winner="FoulPlayBot",
                    pokezero_won=False,
                    decision_rounds=4,
                    pokezero_decisions=4,
                    root_puct_searches=2,
                    root_puct_fallbacks=2,
                    root_puct_total_visits=16,
                    root_puct_effective_total_visits=12,
                    root_puct_opponent_action_scenarios_generated=6,
                    root_puct_opponent_action_scenarios_skipped=3,
                    root_puct_opponent_action_scenarios_unsearched=1,
                    root_puct_opponent_action_skip_categories={
                        "illegal_action_for_current_request": 2,
                        "missing_sampled_world": 1,
                    },
                    root_puct_opponent_action_missing_sampled_world_reason_categories={
                        "opponent_belief_unavailable": 1,
                    },
                    root_puct_opponent_action_replay_rejection_decision_rounds={
                        "12": 2,
                    },
                    root_puct_opponent_action_replay_request_mismatch_decision_rounds={"12": 1},
                    root_puct_opponent_action_replay_request_mismatch_players={
                        "missing:p1": 1,
                        "unexpected:p2": 1,
                    },
                    root_puct_opponent_action_replay_request_mismatch_shapes={
                        "requested:p1|actions:p2": 1,
                    },
                    root_puct_opponent_action_start_override_mismatch_decision_rounds={"12": 1},
                    root_puct_opponent_action_first_observation_mismatch_paths={
                        "numeric_features/opponent_pokemon[8][0]": 1,
                    },
                    root_puct_opponent_action_groups_generated=4,
                    root_puct_opponent_action_groups_used=2,
                    root_puct_opponent_action_groups_skipped=1,
                    root_puct_opponent_action_groups_unsearched=1,
                    root_puct_selected_prior_action_changes=1,
                    root_puct_pre_gate_prior_action_changes=2,
                    root_puct_time_budget_exhaustions=1,
                    root_puct_start_override_sources_used=1,
                    root_puct_start_override_attempts_used=4,
                    root_puct_start_override_duplicate_attempts=2,
                    root_puct_start_override_shared_samples=3,
                    root_puct_start_override_shared_samples_accepted=1,
                    root_puct_start_override_shared_samples_rejected=2,
                    root_puct_fallback_reasons={"search failed: boom": 1, mixed_replay_reason: 1},
                    root_puct_average_elapsed_seconds=0.4,
                ),
            ),
        )

        payload = result.to_dict()

        self.assertEqual(payload["schema_version"], "pokezero.controlled-foulplay-benchmark.v1")
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["complete"], True)
        self.assertEqual(payload["wins"], 1)
        self.assertEqual(payload["completed_games"], 2)
        self.assertEqual(payload["win_rate"], 0.5)
        self.assertEqual(payload["foulplay_random_seed"], 1)
        self.assertEqual(payload["root_puct"]["searches"], 5)
        self.assertEqual(payload["root_puct"]["fallbacks"], 2)
        self.assertEqual(payload["root_puct"]["total_visits"], 40)
        self.assertEqual(payload["root_puct"]["effective_total_visits"], 30)
        self.assertEqual(payload["root_puct"]["opponent_action_scenarios_generated"], 15)
        self.assertEqual(payload["root_puct"]["opponent_action_scenarios_skipped"], 4)
        self.assertEqual(payload["root_puct"]["opponent_action_scenarios_unsearched"], 3)
        self.assertEqual(
            payload["root_puct"]["opponent_action_skip_categories"],
            {
                "illegal_action_for_current_request": 2,
                "missing_sampled_world": 1,
                "start_override_observation_mismatch": 1,
            },
        )
        self.assertEqual(
            payload["root_puct"]["opponent_action_missing_sampled_world_reason_categories"],
            {
                "opponent_belief_unavailable": 1,
                "self_team_unavailable": 2,
            },
        )
        self.assertEqual(
            payload["root_puct"]["opponent_action_replay_rejection_decision_rounds"],
            {"3": 1, "12": 2},
        )
        self.assertEqual(
            payload["root_puct"]["opponent_action_replay_request_mismatch_decision_rounds"],
            {"12": 1},
        )
        self.assertEqual(
            payload["root_puct"]["opponent_action_replay_request_mismatch_players"],
            {"missing:p1": 1, "unexpected:p2": 1},
        )
        self.assertEqual(
            payload["root_puct"]["opponent_action_replay_request_mismatch_shapes"],
            {"requested:p1|actions:p2": 1},
        )
        self.assertEqual(
            payload["root_puct"]["opponent_action_start_override_mismatch_decision_rounds"],
            {"3": 1, "12": 1},
        )
        self.assertEqual(
            payload["root_puct"]["opponent_action_first_observation_mismatch_paths"],
            {
                "categorical_ids/opponent_pokemon[8][11]": 1,
                "numeric_features/opponent_pokemon[8][0]": 1,
            },
        )
        self.assertEqual(payload["root_puct"]["opponent_action_groups_generated"], 9)
        self.assertEqual(payload["root_puct"]["opponent_action_groups_used"], 5)
        self.assertEqual(payload["root_puct"]["opponent_action_groups_skipped"], 2)
        self.assertEqual(payload["root_puct"]["opponent_action_groups_unsearched"], 2)
        self.assertEqual(payload["root_puct"]["selected_prior_action_changes"], 3)
        self.assertEqual(payload["root_puct"]["pre_gate_prior_action_changes"], 5)
        self.assertEqual(payload["root_puct"]["time_budget_exhaustions"], 3)
        self.assertEqual(payload["root_puct"]["start_override_sources_used"], 4)
        self.assertEqual(payload["root_puct"]["start_override_attempts"], 7)
        self.assertEqual(payload["root_puct"]["start_override_attempts_used"], 9)
        self.assertEqual(payload["root_puct"]["start_override_duplicate_attempts"], 3)
        self.assertEqual(payload["root_puct"]["start_override_shared_samples"], 9)
        self.assertEqual(payload["root_puct"]["start_override_shared_samples_accepted"], 5)
        self.assertEqual(payload["root_puct"]["start_override_shared_samples_rejected"], 4)
        self.assertEqual(
            payload["root_puct"]["fallback_reasons"],
            {"search failed: boom": 1, mixed_replay_reason: 1},
        )
        self.assertEqual(
            payload["root_puct"]["fallback_categories"],
            {"mixed_replay_prefix_divergence": 1, "search_failed": 1},
        )
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_scenarios_generated"], 9)
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_scenarios_skipped"], 1)
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_scenarios_unsearched"], 2)
        self.assertEqual(
            payload["game_results"][0]["root_puct_opponent_action_skip_categories"],
            {"start_override_observation_mismatch": 1},
        )
        self.assertEqual(
            payload["game_results"][0]["root_puct_opponent_action_missing_sampled_world_reason_categories"],
            {"self_team_unavailable": 2},
        )
        self.assertEqual(
            payload["game_results"][0]["root_puct_opponent_action_replay_rejection_decision_rounds"],
            {"3": 1},
        )
        self.assertEqual(
            payload["game_results"][0]["root_puct_opponent_action_start_override_mismatch_decision_rounds"],
            {"3": 1},
        )
        self.assertEqual(
            payload["game_results"][0]["root_puct_opponent_action_first_observation_mismatch_paths"],
            {"categorical_ids/opponent_pokemon[8][11]": 1},
        )
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_groups_generated"], 5)
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_groups_used"], 3)
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_groups_skipped"], 1)
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_groups_unsearched"], 1)
        self.assertEqual(payload["game_results"][0]["root_puct_selected_prior_action_changes"], 2)
        self.assertEqual(payload["game_results"][0]["root_puct_pre_gate_prior_action_changes"], 3)
        self.assertEqual(payload["game_results"][0]["root_puct_time_budget_exhaustions"], 2)
        self.assertEqual(payload["game_results"][0]["root_puct_start_override_sources_used"], 3)
        self.assertEqual(payload["game_results"][0]["root_puct_start_override_attempts_used"], 5)
        self.assertEqual(payload["game_results"][0]["root_puct_start_override_duplicate_attempts"], 1)
        self.assertEqual(payload["game_results"][0]["root_puct_start_override_shared_samples"], 6)
        self.assertEqual(payload["game_results"][0]["root_puct_start_override_shared_samples_accepted"], 4)
        self.assertEqual(payload["game_results"][0]["root_puct_start_override_shared_samples_rejected"], 2)
        self.assertEqual(
            payload["game_results"][0]["root_puct_prior_action_change_details"],
            [
                {
                    "decision_index": 1,
                    "selected_action": 4,
                    "search_action": 4,
                    "prior_action": 0,
                    "selected_changed_prior_action": True,
                    "pre_gate_changed_prior_action": True,
                },
            ],
        )
        self.assertEqual(
            payload["game_results"][1]["root_puct_fallback_reasons"],
            {"search failed: boom": 1, mixed_replay_reason: 1},
        )
        self.assertEqual(
            payload["game_results"][1]["root_puct_fallback_categories"],
            {"mixed_replay_prefix_divergence": 1, "search_failed": 1},
        )
        self.assertEqual(
            payload["game_results"][1]["root_puct_opponent_action_skip_categories"],
            {
                "illegal_action_for_current_request": 2,
                "missing_sampled_world": 1,
            },
        )
        self.assertEqual(
            payload["game_results"][1]["root_puct_opponent_action_missing_sampled_world_reason_categories"],
            {"opponent_belief_unavailable": 1},
        )
        self.assertEqual(
            payload["game_results"][1]["root_puct_opponent_action_replay_request_mismatch_players"],
            {"missing:p1": 1, "unexpected:p2": 1},
        )
        self.assertEqual(
            payload["game_results"][1]["root_puct_opponent_action_replay_request_mismatch_shapes"],
            {"requested:p1|actions:p2": 1},
        )
        self.assertEqual(payload["root_puct"]["opponent_legal_mask_mode"], "hidden")
        self.assertEqual(payload["root_puct"]["foulplay_search_time_ms"], 1000)
        self.assertEqual(payload["root_puct"]["selection_mode"], "visits")
        self.assertEqual(payload["root_puct"]["minimum_value_improvement"], 0.25)
        self.assertEqual(payload["root_puct"]["minimum_override_prior_ratio"], 0.5)
        self.assertEqual(payload["root_puct"]["minimum_score_improvement"], 0.1)
        self.assertEqual(payload["root_puct"]["root_prior_temperature"], 2.5)
        self.assertIsNone(payload["root_puct"]["root_visit_budget"])
        self.assertEqual(payload["root_puct"]["root_time_budget_ms"], 250)
        self.assertEqual(payload["root_puct"]["root_opponent_action_scenarios"], 2)
        self.assertEqual(payload["root_puct"]["root_opponent_action_candidate_scenarios"], 5)
        self.assertEqual(payload["root_puct"]["leaf_rollout_sampling"], True)
        self.assertEqual(payload["root_puct"]["belief_start_overrides"], True)
        self.assertEqual(payload["root_puct"]["belief_start_override_samples"], 3)
        self.assertAlmostEqual(payload["root_puct"]["average_elapsed_seconds"], 0.3)

    def test_comparison_payload_matches_common_seeds_and_marks_small_samples_diagnostic(self) -> None:
        raw_config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=3,
            seed_start=10,
            policy_mode="raw",
        )
        search_config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=3,
            seed_start=10,
            policy_mode="root-puct",
        )
        raw = ControlledFoulPlayBenchmarkResult(
            config=raw_config,
            policy_id="checkpoint",
            games=(
                ControlledFoulPlayGameResult(
                    battle_id="battle-10",
                    seed=10,
                    winner="PokeZeroBot",
                    pokezero_won=True,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=0,
                    root_puct_fallbacks=0,
                ),
                ControlledFoulPlayGameResult(
                    battle_id="battle-11",
                    seed=11,
                    winner="FoulPlayBot",
                    pokezero_won=False,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=0,
                    root_puct_fallbacks=0,
                ),
                ControlledFoulPlayGameResult(
                    battle_id="battle-12",
                    seed=12,
                    winner="FoulPlayBot",
                    pokezero_won=False,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=0,
                    root_puct_fallbacks=0,
                ),
            ),
        )
        search = ControlledFoulPlayBenchmarkResult(
            config=search_config,
            policy_id="checkpoint+root-puct",
            games=(
                ControlledFoulPlayGameResult(
                    battle_id="battle-11",
                    seed=11,
                    winner="PokeZeroBot",
                    pokezero_won=True,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=1,
                    root_puct_fallbacks=0,
                ),
                ControlledFoulPlayGameResult(
                    battle_id="battle-12",
                    seed=12,
                    winner="FoulPlayBot",
                    pokezero_won=False,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=1,
                    root_puct_fallbacks=0,
                ),
            ),
        )
        comparison = ControlledFoulPlayComparisonResult(
            config=search_config,
            raw=raw,
            root_puct=search,
        )

        payload = comparison.to_dict()

        self.assertEqual(payload["schema_version"], "pokezero.controlled-foulplay-comparison.v1")
        self.assertEqual(payload["status"], "partial")
        self.assertFalse(payload["complete"])
        self.assertEqual(payload["runs"]["raw"]["policy_mode"], "raw")
        self.assertEqual(payload["runs"]["root_puct"]["policy_mode"], "root-puct")
        self.assertEqual(payload["comparison_mode"], "per-seed")
        self.assertEqual(payload["comparison"]["sample_size"]["status"], "diagnostic_only")
        self.assertEqual(payload["comparison"]["sample_size"]["paired_games"], 2)
        self.assertEqual(payload["comparison"]["sample_size"]["minimum_strength_games"], 300)
        self.assertEqual(payload["comparison"]["aggregate"]["raw"]["wins"], 1)
        self.assertEqual(payload["comparison"]["aggregate"]["raw"]["games"], 3)
        self.assertAlmostEqual(payload["comparison"]["aggregate"]["raw"]["win_rate"], 1 / 3)
        self.assertEqual(payload["comparison"]["paired_by_seed"]["games"], 2)
        self.assertEqual(
            payload["comparison"]["paired_by_seed"]["pairing_method"],
            "per_seed_shared_battlestream_seed_and_foulplay_start_seed",
        )
        self.assertEqual(payload["comparison"]["paired_by_seed"]["opponent_deterministic"], False)
        self.assertEqual(payload["comparison"]["paired_by_seed"]["paired_counterfactual"], False)
        self.assertEqual(
            payload["comparison"]["paired_by_seed"]["interval_method"],
            "marginal_wilson_per_arm_not_paired_delta",
        )
        self.assertEqual(payload["comparison"]["paired_by_seed"]["delta_interpretation"], "descriptive_only")
        self.assertEqual(payload["comparison"]["paired_by_seed"]["raw"]["wins"], 0)
        self.assertEqual(payload["comparison"]["paired_by_seed"]["root_puct"]["wins"], 1)
        self.assertEqual(payload["comparison"]["paired_by_seed"]["raw"]["interval_method"], "wilson_score_marginal_95")
        self.assertEqual(
            payload["comparison"]["paired_by_seed"]["discordant_pairs"],
            {
                "both_won": 0,
                "raw_only_won": 0,
                "root_puct_only_won": 1,
                "neither_won": 1,
            },
        )
        self.assertEqual(payload["comparison"]["paired_by_seed"]["first_seed"], 11)
        self.assertEqual(payload["comparison"]["paired_by_seed"]["last_seed"], 12)
        self.assertAlmostEqual(payload["comparison"]["paired_by_seed"]["root_puct_minus_raw_win_rate"], 0.5)
        self.assertIsNotNone(payload["comparison"]["paired_by_seed"]["root_puct"]["wilson_95"])

    def test_run_controlled_foulplay_comparison_forces_raw_then_root_puct(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=2,
            policy_mode="raw",
        )
        observed_modes: list[str] = []
        observed_seed_starts: list[int] = []
        observed_foulplay_random_seeds: list[int] = []
        progress_payloads: list[dict[str, object]] = []

        async def fake_benchmark(
            benchmark_config: ControlledFoulPlayConfig,
            *,
            progress_callback=None,
        ) -> ControlledFoulPlayBenchmarkResult:
            observed_modes.append(benchmark_config.policy_mode)
            observed_seed_starts.append(benchmark_config.seed_start)
            observed_foulplay_random_seeds.append(benchmark_config.resolved_foulplay_random_seed)
            result = ControlledFoulPlayBenchmarkResult(
                config=benchmark_config,
                policy_id=f"checkpoint-{benchmark_config.policy_mode}",
                games=(
                    ControlledFoulPlayGameResult(
                        battle_id=f"battle-{benchmark_config.policy_mode}",
                        seed=benchmark_config.seed_start,
                        winner="PokeZeroBot" if benchmark_config.policy_mode == "root-puct" else "FoulPlayBot",
                        pokezero_won=benchmark_config.policy_mode == "root-puct",
                        decision_rounds=1,
                        pokezero_decisions=1,
                        root_puct_searches=1 if benchmark_config.policy_mode == "root-puct" else 0,
                        root_puct_fallbacks=0,
                    ),
                ),
            )
            if progress_callback is not None:
                progress_callback(result)
            return result

        with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", side_effect=fake_benchmark):
            comparison = asyncio.run(
                run_controlled_foulplay_comparison(
                    config,
                    progress_callback=lambda result: progress_payloads.append(result.to_dict()),
                )
            )

        self.assertEqual(observed_modes, ["raw", "root-puct", "raw", "root-puct"])
        self.assertEqual(observed_seed_starts, [1, 1, 2, 2])
        self.assertEqual(observed_foulplay_random_seeds, [1, 1, 2, 2])
        self.assertEqual(comparison.raw.config.policy_mode, "raw")
        self.assertEqual(comparison.root_puct.config.policy_mode, "root-puct")
        self.assertEqual(comparison.raw.completed_games, 2)
        self.assertEqual(comparison.root_puct.completed_games, 2)
        self.assertEqual(progress_payloads[0]["runs"]["root_puct"], None)
        self.assertIsNone(progress_payloads[0]["comparison"]["aggregate"]["root_puct_minus_raw_win_rate"])
        self.assertIsNone(progress_payloads[0]["comparison"]["paired_by_seed"]["root_puct_minus_raw_win_rate"])
        self.assertEqual(progress_payloads[1]["comparison"]["paired_by_seed"]["games"], 1)
        self.assertEqual(progress_payloads[1]["comparison"]["paired_by_seed"]["root_puct_minus_raw_win_rate"], 1.0)
        payload = comparison.to_dict()
        self.assertEqual(payload["comparison"]["paired_by_seed"]["root_puct"]["wins"], 2)
        self.assertEqual(payload["foulplay_random_seed_schedule"]["seeds"], [1, 2])
        self.assertEqual(payload["runs"]["raw"]["foulplay_random_seed_schedule"]["seeds"], [1, 2])
        self.assertEqual(payload["runs"]["root_puct"]["foulplay_random_seed_schedule"]["seeds"], [1, 2])

    def test_run_controlled_foulplay_comparison_records_explicit_foulplay_seed_schedule(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=2,
            seed_start=11,
            foulplay_random_seed=456,
        )
        observed: list[tuple[str, int, int]] = []

        async def fake_benchmark(
            benchmark_config: ControlledFoulPlayConfig,
            *,
            progress_callback=None,
        ) -> ControlledFoulPlayBenchmarkResult:
            observed.append(
                (
                    benchmark_config.policy_mode,
                    benchmark_config.seed_start,
                    benchmark_config.resolved_foulplay_random_seed,
                )
            )
            result = ControlledFoulPlayBenchmarkResult(
                config=benchmark_config,
                policy_id=f"checkpoint-{benchmark_config.policy_mode}",
                games=(
                    ControlledFoulPlayGameResult(
                        battle_id=f"battle-{benchmark_config.policy_mode}-{benchmark_config.seed_start}",
                        seed=benchmark_config.seed_start,
                        winner="FoulPlayBot",
                        pokezero_won=False,
                        decision_rounds=1,
                        pokezero_decisions=1,
                        root_puct_searches=1 if benchmark_config.policy_mode == "root-puct" else 0,
                        root_puct_fallbacks=0,
                    ),
                ),
            )
            if progress_callback is not None:
                progress_callback(result)
            return result

        with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", side_effect=fake_benchmark):
            comparison = asyncio.run(run_controlled_foulplay_comparison(config))

        self.assertEqual(
            observed,
            [
                ("raw", 11, 456),
                ("root-puct", 11, 456),
                ("raw", 12, 457),
                ("root-puct", 12, 457),
            ],
        )
        payload = comparison.to_dict()
        self.assertEqual(payload["foulplay_random_seed"], 456)
        self.assertEqual(payload["foulplay_random_seed_schedule"]["seeds"], [456, 457])
        self.assertEqual(payload["runs"]["raw"]["foulplay_random_seed_schedule"]["seeds"], [456, 457])
        self.assertEqual(payload["runs"]["root_puct"]["foulplay_random_seed_schedule"]["seeds"], [456, 457])

    def test_per_seed_comparison_skips_seed_and_records_crash_when_foulplay_exits_early(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=3,
            policy_mode="raw",
            opponent_crash_retries=0,
        )
        observed_arms: list[tuple[str, int]] = []

        async def fake_benchmark(
            benchmark_config: ControlledFoulPlayConfig,
            *,
            progress_callback=None,
        ) -> ControlledFoulPlayBenchmarkResult:
            observed_arms.append((benchmark_config.policy_mode, benchmark_config.seed_start))
            if benchmark_config.policy_mode == "root-puct" and benchmark_config.seed_start == 2:
                raise FoulPlayProcessExitError(
                    stage="choosing",
                    returncode=1,
                    log_tail="stderr:\n_pickle.PicklingError: Can't pickle pyo3_runtime.PanicException",
                )
            return ControlledFoulPlayBenchmarkResult(
                config=benchmark_config,
                policy_id=f"checkpoint-{benchmark_config.policy_mode}",
                games=(
                    ControlledFoulPlayGameResult(
                        battle_id=f"battle-{benchmark_config.policy_mode}-{benchmark_config.seed_start}",
                        seed=benchmark_config.seed_start,
                        winner="PokeZeroBot",
                        pokezero_won=True,
                        decision_rounds=1,
                        pokezero_decisions=1,
                        root_puct_searches=0,
                        root_puct_fallbacks=0,
                    ),
                ),
            )

        with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", side_effect=fake_benchmark):
            comparison = asyncio.run(run_controlled_foulplay_comparison(config))

        self.assertEqual(
            observed_arms,
            [
                ("raw", 1),
                ("root-puct", 1),
                ("raw", 2),
                ("root-puct", 2),
                ("raw", 3),
                ("root-puct", 3),
            ],
        )
        self.assertEqual([game.seed for game in comparison.raw.games], [1, 3])
        self.assertEqual([game.seed for game in comparison.root_puct.games], [1, 3])
        self.assertTrue(comparison.complete)
        payload = comparison.to_dict()
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["runs"]["raw"]["foulplay_random_seed_schedule"]["seeds"], [1, 3])
        self.assertEqual(payload["runs"]["root_puct"]["foulplay_random_seed_schedule"]["seeds"], [1, 3])
        self.assertEqual(payload["comparison"]["paired_by_seed"]["games"], 2)
        self.assertEqual(
            payload["comparison"]["opponent_crashed_seeds"],
            {
                "count": 1,
                "seeds": [2],
                "handling": "seed_excluded_from_paired_stats_and_aggregates",
            },
        )
        self.assertEqual(len(payload["opponent_crashes"]), 1)
        crash = payload["opponent_crashes"][0]
        self.assertEqual(crash["seed"], 2)
        self.assertEqual(crash["policy_mode"], "root-puct")
        self.assertEqual(crash["returncode"], 1)
        self.assertEqual(crash["attempts"], 1)
        self.assertEqual(crash["stage"], "choosing")
        self.assertIn("PanicException", crash["stderr_tail"])

    def test_per_seed_comparison_retries_crashed_arm_once_by_default(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=1,
            policy_mode="raw",
        )
        raw_attempts = 0

        async def fake_benchmark(
            benchmark_config: ControlledFoulPlayConfig,
            *,
            progress_callback=None,
        ) -> ControlledFoulPlayBenchmarkResult:
            nonlocal raw_attempts
            if benchmark_config.policy_mode == "raw":
                raw_attempts += 1
                if raw_attempts == 1:
                    raise FoulPlayProcessExitError(stage="challenging", returncode=2, log_tail="stderr:\nboom")
            return ControlledFoulPlayBenchmarkResult(
                config=benchmark_config,
                policy_id=f"checkpoint-{benchmark_config.policy_mode}",
                games=(
                    ControlledFoulPlayGameResult(
                        battle_id=f"battle-{benchmark_config.policy_mode}",
                        seed=benchmark_config.seed_start,
                        winner="PokeZeroBot",
                        pokezero_won=True,
                        decision_rounds=1,
                        pokezero_decisions=1,
                        root_puct_searches=0,
                        root_puct_fallbacks=0,
                    ),
                ),
            )

        with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", side_effect=fake_benchmark):
            comparison = asyncio.run(run_controlled_foulplay_comparison(config))

        self.assertEqual(raw_attempts, 2)
        self.assertEqual(comparison.opponent_crashes, ())
        self.assertEqual(comparison.raw.completed_games, 1)
        self.assertEqual(comparison.root_puct.completed_games, 1)
        self.assertTrue(comparison.complete)

    def test_per_seed_comparison_skips_root_puct_arm_when_raw_arm_crashes(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=2,
            policy_mode="raw",
            opponent_crash_retries=0,
        )
        observed_arms: list[tuple[str, int]] = []

        async def fake_benchmark(
            benchmark_config: ControlledFoulPlayConfig,
            *,
            progress_callback=None,
        ) -> ControlledFoulPlayBenchmarkResult:
            observed_arms.append((benchmark_config.policy_mode, benchmark_config.seed_start))
            if benchmark_config.policy_mode == "raw" and benchmark_config.seed_start == 1:
                raise FoulPlayProcessExitError(stage="choosing", returncode=137, log_tail="stderr:\nkilled")
            return ControlledFoulPlayBenchmarkResult(
                config=benchmark_config,
                policy_id=f"checkpoint-{benchmark_config.policy_mode}",
                games=(
                    ControlledFoulPlayGameResult(
                        battle_id=f"battle-{benchmark_config.policy_mode}-{benchmark_config.seed_start}",
                        seed=benchmark_config.seed_start,
                        winner="FoulPlayBot",
                        pokezero_won=False,
                        decision_rounds=1,
                        pokezero_decisions=1,
                        root_puct_searches=0,
                        root_puct_fallbacks=0,
                    ),
                ),
            )

        with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", side_effect=fake_benchmark):
            comparison = asyncio.run(run_controlled_foulplay_comparison(config))

        self.assertEqual(observed_arms, [("raw", 1), ("raw", 2), ("root-puct", 2)])
        self.assertEqual([game.seed for game in comparison.raw.games], [2])
        self.assertEqual([game.seed for game in comparison.root_puct.games], [2])
        self.assertEqual(len(comparison.opponent_crashes), 1)
        self.assertEqual(comparison.opponent_crashes[0].seed, 1)
        self.assertEqual(comparison.opponent_crashes[0].policy_mode, "raw")
        self.assertEqual(comparison.opponent_crashes[0].returncode, 137)
        self.assertTrue(comparison.complete)

    def test_run_controlled_foulplay_comparison_can_preserve_per_arm_order(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=2,
        )
        observed_modes: list[str] = []

        async def fake_benchmark(
            benchmark_config: ControlledFoulPlayConfig,
            *,
            progress_callback=None,
        ) -> ControlledFoulPlayBenchmarkResult:
            observed_modes.append(benchmark_config.policy_mode)
            result = ControlledFoulPlayBenchmarkResult(
                config=benchmark_config,
                policy_id=f"checkpoint-{benchmark_config.policy_mode}",
                games=(
                    ControlledFoulPlayGameResult(
                        battle_id=f"battle-{benchmark_config.policy_mode}",
                        seed=benchmark_config.seed_start,
                        winner="PokeZeroBot" if benchmark_config.policy_mode == "root-puct" else "FoulPlayBot",
                        pokezero_won=benchmark_config.policy_mode == "root-puct",
                        decision_rounds=1,
                        pokezero_decisions=1,
                        root_puct_searches=1 if benchmark_config.policy_mode == "root-puct" else 0,
                        root_puct_fallbacks=0,
                    ),
                ),
            )
            if progress_callback is not None:
                progress_callback(result)
            return result

        with patch("pokezero.foulplay_bridge.run_controlled_foulplay_benchmark", side_effect=fake_benchmark):
            comparison = asyncio.run(
                run_controlled_foulplay_comparison(
                    config,
                    comparison_mode="per-arm",
                )
            )

        self.assertEqual(observed_modes, ["raw", "root-puct"])
        self.assertEqual(comparison.comparison_mode, "per-arm")
        self.assertEqual(
            comparison.to_dict()["comparison"]["paired_by_seed"]["pairing_method"],
            "shared_battlestream_seed_only",
        )

    def test_comparison_cli_writes_summary_out(self) -> None:
        parser_help = build_comparison_arg_parser().format_help()
        self.assertNotIn("--policy-mode", parser_help)

        async def fake_comparison(
            config: ControlledFoulPlayConfig,
            *,
            comparison_mode="per-seed",
            progress_callback=None,
        ) -> ControlledFoulPlayComparisonResult:
            raw = ControlledFoulPlayBenchmarkResult(
                config=ControlledFoulPlayConfig(
                    checkpoint=config.checkpoint,
                    showdown_root=config.showdown_root,
                    games=config.games,
                    seed_start=config.seed_start,
                    policy_mode="raw",
                ),
                policy_id="checkpoint",
                games=(
                    ControlledFoulPlayGameResult(
                        battle_id="battle-1",
                        seed=config.seed_start,
                        winner="FoulPlayBot",
                        pokezero_won=False,
                        decision_rounds=1,
                        pokezero_decisions=1,
                        root_puct_searches=0,
                        root_puct_fallbacks=0,
                    ),
                ),
            )
            search = ControlledFoulPlayBenchmarkResult(
                config=ControlledFoulPlayConfig(
                    checkpoint=config.checkpoint,
                    showdown_root=config.showdown_root,
                    games=config.games,
                    seed_start=config.seed_start,
                    policy_mode="root-puct",
                ),
                policy_id="checkpoint+root-puct",
                games=(
                    ControlledFoulPlayGameResult(
                        battle_id="battle-1",
                        seed=config.seed_start,
                        winner="PokeZeroBot",
                        pokezero_won=True,
                        decision_rounds=1,
                        pokezero_decisions=1,
                        root_puct_searches=1,
                        root_puct_fallbacks=0,
                    ),
                ),
            )
            result = ControlledFoulPlayComparisonResult(
                config=config,
                raw=raw,
                root_puct=search,
                comparison_mode=comparison_mode,
            )
            if progress_callback is not None:
                progress_callback(result)
            return result

        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "comparison.json"
            argv = (
                "--checkpoint",
                "checkpoint.pt",
                "--showdown-root",
                "/showdown",
                "--games",
                "1",
                "--max-decision-rounds",
                "17",
                "--progress-interval-games",
                "1",
                "--summary-out",
                str(summary_path),
            )
            with patch(
                "pokezero.foulplay_bridge.run_controlled_foulplay_comparison",
                side_effect=fake_comparison,
            ), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch(
                "sys.stderr", new_callable=io.StringIO
            ) as stderr:
                exit_code = asyncio.run(async_comparison_main(argv))

            payload = json.loads(summary_path.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema_version"], "pokezero.controlled-foulplay-comparison.v1")
        self.assertEqual(payload["comparison_mode"], "per-seed")
        self.assertEqual(payload["max_decision_rounds"], 17)
        self.assertEqual(payload["comparison"]["paired_by_seed"]["root_puct"]["wins"], 1)
        self.assertEqual(build_comparison_arg_parser().parse_args(argv).games, 1)
        self.assertEqual(build_comparison_arg_parser().parse_args(argv).comparison_mode, "per-seed")
        self.assertEqual(build_comparison_arg_parser().parse_args(argv).progress_interval_games, 1)
        self.assertIn("DIAGNOSTIC RESULT", stdout.getvalue())
        self.assertIn("(per-seed)", stdout.getvalue())
        self.assertIn("descriptive_delta=100.0%", stdout.getvalue())
        progress_lines = [
            line
            for line in stderr.getvalue().splitlines()
            if line.startswith("controlled_foulplay_comparison_progress:")
        ]
        self.assertEqual(
            [json.loads(line.split(": ", 1)[1]) for line in progress_lines],
            [
                {
                    "comparison_mode": "per-seed",
                    "games_completed": 1,
                    "games_total": 1,
                    "opponent_crash_count": 0,
                }
            ],
        )

    def test_comparison_progress_reports_interval_and_final_state(self) -> None:
        emit_progress = _controlled_foulplay_comparison_progress_callback(2)

        def result(completed_games: int) -> SimpleNamespace:
            return SimpleNamespace(
                raw=SimpleNamespace(completed_games=completed_games),
                root_puct=SimpleNamespace(completed_games=completed_games),
                config=SimpleNamespace(games=3),
                comparison_mode="per-seed",
                opponent_crashes=(),
            )

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            emit_progress(result(1))
            emit_progress(result(2))
            emit_progress(result(3), force=True)

        lines = [
            json.loads(line.split(": ", 1)[1])
            for line in stderr.getvalue().splitlines()
            if line.startswith("controlled_foulplay_comparison_progress:")
        ]
        self.assertEqual([line["games_completed"] for line in lines], [2, 3])

    def test_observation_with_search_metadata_adds_belief_view_without_mutating_original(self) -> None:
        class BeliefView:
            def to_overlay_payload(self):
                return {"self_slot": "p1", "opponent_slot": "p2"}

        class State:
            belief_view = BeliefView()

        observation = PokeZeroObservationV0(
            categorical_ids=(),
            numeric_features=(),
            token_type_ids=(),
            attention_mask=(),
            legal_action_mask=(True,) + (False,) * 8,
            metadata={"existing": "value"},
        )

        augmented = _observation_with_search_metadata(observation, State())  # type: ignore[arg-type]

        self.assertNotIn("belief_view", observation.metadata)
        self.assertEqual(augmented.metadata["existing"], "value")
        self.assertEqual(augmented.metadata["belief_view"]["self_slot"], "p1")

    def test_write_json_creates_parent_directory_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "summary.json"

            _write_json(path, {"b": 2, "a": 1})

            self.assertEqual(path.read_text(), '{\n  "a": 1,\n  "b": 2\n}\n')

    def test_root_puct_prior_action_change_details_extracts_changed_non_fallback_decisions(self) -> None:
        decisions = (
            PolicyDecision(
                action_index=0,
                policy_id="root-puct",
                metadata={
                    "policy_family": "root-puct-search",
                    "root_puct_fallback": False,
                    "root_puct_selected_changed_prior_action": False,
                    "root_puct_pre_gate_changed_prior_action": False,
                },
            ),
            PolicyDecision(
                action_index=4,
                policy_id="root-puct",
                metadata={
                    "policy_family": "root-puct-search",
                    "root_puct_fallback": False,
                    "root_puct_selected_changed_prior_action": True,
                    "root_puct_pre_gate_changed_prior_action": True,
                    "root_puct_search_action": 4,
                    "root_puct_prior_action": 0,
                    "root_puct_selected_value": 0.25,
                    "root_puct_search_action_value": 0.25,
                    "root_puct_prior_value": 0.1,
                    "root_puct_selected_score": 1.5,
                    "root_puct_search_action_score": 1.5,
                    "root_puct_prior_score": 0.9,
                    "root_puct_selected_action_prior": 0.2,
                    "root_puct_search_action_prior": 0.2,
                    "root_puct_prior_action_prior": 0.8,
                    "root_puct_selected_action_visits": 4,
                    "root_puct_search_action_visits": 4,
                    "root_puct_prior_action_visits": 2,
                },
            ),
            PolicyDecision(
                action_index=2,
                policy_id="root-puct",
                metadata={
                    "policy_family": "root-puct-search",
                    "root_puct_fallback": False,
                    "root_puct_selected_changed_prior_action": False,
                    "root_puct_pre_gate_changed_prior_action": True,
                    "root_puct_value_gate_used": True,
                    "root_puct_search_action": 4,
                    "root_puct_prior_action": 2,
                    "root_puct_selected_value": 0.1,
                    "root_puct_search_action_value": 0.2,
                    "root_puct_prior_value": 0.1,
                    "root_puct_selected_score": 0.8,
                    "root_puct_search_action_score": 0.9,
                    "root_puct_prior_score": 0.8,
                    "root_puct_selected_action_prior": 0.7,
                    "root_puct_search_action_prior": 0.3,
                    "root_puct_prior_action_prior": 0.7,
                    "root_puct_selected_action_visits": 3,
                    "root_puct_search_action_visits": 4,
                    "root_puct_prior_action_visits": 3,
                },
            ),
            PolicyDecision(
                action_index=1,
                policy_id="root-puct",
                metadata={
                    "policy_family": "root-puct-search",
                    "root_puct_fallback": False,
                    "root_puct_selected_changed_prior_action": False,
                    "root_puct_pre_gate_changed_prior_action": True,
                    "root_puct_prior_ratio_gate_used": True,
                    "root_puct_minimum_override_prior_ratio": 0.5,
                    "root_puct_prior_ratio_gate_required_prior": 0.35,
                    "root_puct_score_gate_used": True,
                    "root_puct_minimum_score_improvement": 0.1,
                    "root_puct_score_gate_required_score": 0.8,
                    "root_puct_search_action": 0,
                    "root_puct_prior_action": 1,
                    "root_puct_selected_value": 0.1,
                    "root_puct_search_action_value": 0.3,
                    "root_puct_prior_value": 0.1,
                    "root_puct_selected_score": 0.7,
                    "root_puct_search_action_score": 0.8,
                    "root_puct_prior_score": 0.7,
                    "root_puct_selected_action_prior": 0.7,
                    "root_puct_search_action_prior": 0.2,
                    "root_puct_prior_action_prior": 0.7,
                    "root_puct_selected_action_visits": 3,
                    "root_puct_search_action_visits": 4,
                    "root_puct_prior_action_visits": 3,
                },
            ),
            PolicyDecision(
                action_index=2,
                policy_id="root-puct",
                metadata={
                    "policy_family": "root-puct-search",
                    "root_puct_fallback": True,
                    "root_puct_selected_changed_prior_action": True,
                    "root_puct_search_action": 2,
                    "root_puct_prior_action": 0,
                },
            ),
        )

        details = _root_puct_prior_action_change_details(decisions)

        self.assertEqual(len(details), 3)
        self.assertEqual(details[0]["decision_index"], 1)
        self.assertEqual(details[0]["selected_action"], 4)
        self.assertEqual(details[0]["search_action"], 4)
        self.assertEqual(details[0]["prior_action"], 0)
        self.assertEqual(details[0]["selected_value"], 0.25)
        self.assertEqual(details[0]["prior_action_prior"], 0.8)
        self.assertEqual(details[0]["selected_visits"], 4)
        self.assertFalse(details[0]["value_gate_used"])
        self.assertEqual(details[1]["decision_index"], 2)
        self.assertEqual(details[1]["selected_action"], 2)
        self.assertEqual(details[1]["search_action"], 4)
        self.assertEqual(details[1]["prior_action"], 2)
        self.assertFalse(details[1]["selected_changed_prior_action"])
        self.assertTrue(details[1]["pre_gate_changed_prior_action"])
        self.assertTrue(details[1]["value_gate_used"])
        self.assertEqual(details[1]["selected_action_prior"], 0.7)
        self.assertEqual(details[1]["search_action_prior"], 0.3)
        self.assertEqual(details[2]["decision_index"], 3)
        self.assertEqual(details[2]["selected_action"], 1)
        self.assertEqual(details[2]["search_action"], 0)
        self.assertEqual(details[2]["prior_action"], 1)
        self.assertFalse(details[2]["selected_changed_prior_action"])
        self.assertTrue(details[2]["pre_gate_changed_prior_action"])
        self.assertFalse(details[2]["value_gate_used"])
        self.assertTrue(details[2]["prior_ratio_gate_used"])
        self.assertEqual(details[2]["minimum_override_prior_ratio"], 0.5)
        self.assertEqual(details[2]["prior_ratio_gate_required_prior"], 0.35)
        self.assertTrue(details[2]["score_gate_used"])
        self.assertEqual(details[2]["minimum_score_improvement"], 0.1)
        self.assertEqual(details[2]["score_gate_required_score"], 0.8)

    def test_run_controlled_foulplay_benchmark_emits_incremental_progress(self) -> None:
        class FakeModelConfig:
            policy_id = "checkpoint"
            observation_schema_version = "pokezero.observation.v2.1"
            categorical_feature_count = 1
            numeric_feature_count = 1
            stats_block_enabled = True
            exact_state_enabled = True
            transition_token_budget = 128
            # Region-trim plumbing added transition_token_count to the model
            # config; observation_spec_from_model_config reads it.
            transition_token_count = 128
            tier2_residuals = True
            tier2_investment = False

        class FakeCheckpointResult:
            model_config = FakeModelConfig()

        class FakePolicy:
            policy_id = "checkpoint+root-puct"

        class FakeProcess:
            stdout = None
            stderr = None

            def __init__(self) -> None:
                self.returncode: int | None = None

            def terminate(self) -> None:
                self.returncode = -15

            async def wait(self) -> int:
                self.returncode = 0
                return self.returncode

        class FakeServer:
            def __init__(self, **_: object) -> None:
                self.uri = "ws://127.0.0.1:1/showdown/websocket"

            async def start(self) -> None:
                return None

            async def close(self) -> None:
                return None

        class FakeBridge:
            def __init__(self, **_: object) -> None:
                return None

            async def start(self) -> None:
                return None

            async def close(self) -> None:
                return None

        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("showdown"),
            games=2,
        )
        game_results = iter(
            (
                ControlledFoulPlayGameResult(
                    battle_id="battle-1",
                    seed=1,
                    winner="PokeZeroBot",
                    pokezero_won=True,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=1,
                    root_puct_fallbacks=0,
                ),
                ControlledFoulPlayGameResult(
                    battle_id="battle-2",
                    seed=2,
                    winner="FoulPlayBot",
                    pokezero_won=False,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=1,
                    root_puct_fallbacks=0,
                ),
            )
        )
        progress_payloads: list[dict[str, object]] = []

        async def wait_for_challenge(**_: object) -> None:
            return None

        async def run_single_game(**kwargs: object) -> ControlledFoulPlayGameResult:
            # Mirrors foul-play exiting during its post-game websocket cleanup.
            process = kwargs["foulplay_process"]
            assert isinstance(process, FakeProcess)
            process.returncode = 0
            return next(game_results)

        spawned_run_counts: list[int | None] = []
        spawned_foulplay_seeds: list[int | None] = []

        async def spawn_foulplay(
            spawned_config: ControlledFoulPlayConfig,
            *_: object,
            run_count: int | None = None,
            **__: object,
        ) -> FakeProcess:
            spawned_run_counts.append(run_count)
            spawned_foulplay_seeds.append(spawned_config.foulplay_random_seed)
            return FakeProcess()

        with (
            patch("pokezero.foulplay_bridge._validate_external_paths"),
            patch("pokezero.foulplay_bridge.load_transformer_checkpoint", return_value=(object(), FakeCheckpointResult())),
            patch("pokezero.foulplay_bridge.category_vocab_from_model_config", return_value=object()),
            patch("pokezero.foulplay_bridge.load_showdown_dex_cached", return_value=object()),
            patch("pokezero.foulplay_bridge._build_policy", return_value=FakePolicy()),
            patch("pokezero.foulplay_bridge._FoulPlayWebsocketServer", FakeServer),
            patch("pokezero.foulplay_bridge._BattleBridge", FakeBridge),
            patch("pokezero.foulplay_bridge._spawn_foulplay", side_effect=spawn_foulplay),
            patch("pokezero.foulplay_bridge._wait_for_foulplay_challenge_or_exit", side_effect=wait_for_challenge),
            patch("pokezero.foulplay_bridge._run_single_game", side_effect=run_single_game),
        ):
            result = asyncio.run(
                run_controlled_foulplay_benchmark(
                    config,
                    progress_callback=lambda partial: progress_payloads.append(partial.to_dict()),
                )
            )

        self.assertEqual(result.completed_games, 2)
        self.assertEqual([payload["completed_games"] for payload in progress_payloads], [1, 2])
        self.assertEqual([payload["status"] for payload in progress_payloads], ["partial", "complete"])
        self.assertEqual([payload["complete"] for payload in progress_payloads], [False, True])
        self.assertEqual(spawned_run_counts, [1, 1])
        self.assertEqual(spawned_foulplay_seeds, [1, 2])
        self.assertEqual(result.to_dict()["foulplay_random_seed_schedule"]["seeds"], [1, 2])

    def test_stale_foulplay_connection_cannot_clear_successor_socket(self) -> None:
        class FakeSocket:
            def __init__(self) -> None:
                self.sent: list[str] = []
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def send(self, message: str) -> None:
                self.sent.append(message)
                self.entered.set()

            def __aiter__(self) -> "FakeSocket":
                return self

            async def __anext__(self) -> str:
                await self.release.wait()
                raise RuntimeError("simulated disconnect")

        async def exercise() -> None:
            server = _FoulPlayWebsocketServer(username="FoulPlayBot", host="127.0.0.1")
            old_socket = FakeSocket()
            new_socket = FakeSocket()
            old_task = asyncio.create_task(server._handle_connection(old_socket))
            await old_socket.entered.wait()
            new_task = asyncio.create_task(server._handle_connection(new_socket))
            await new_socket.entered.wait()

            old_socket.release.set()
            await old_task
            self.assertIs(server.websocket, new_socket)

            new_socket.release.set()
            await new_task
            self.assertIsNone(server.websocket)

        asyncio.run(exercise())

    def test_async_main_summary_out_preserves_partial_progress_on_failure(self) -> None:
        class FakeModelConfig:
            policy_id = "checkpoint"
            observation_schema_version = "pokezero.observation.v2.1"
            categorical_feature_count = 1
            numeric_feature_count = 1
            stats_block_enabled = True
            exact_state_enabled = True
            transition_token_budget = 128
            # Region-trim plumbing added transition_token_count to the model
            # config; observation_spec_from_model_config reads it.
            transition_token_count = 128
            tier2_residuals = True
            tier2_investment = False

        class FakeCheckpointResult:
            model_config = FakeModelConfig()

        class FakePolicy:
            policy_id = "checkpoint+root-puct"

        class FakeProcess:
            stdout = None
            stderr = None
            returncode = 0

            def terminate(self) -> None:
                raise AssertionError("completed fake process should not be terminated")

        class FakeServer:
            def __init__(self, **_: object) -> None:
                self.uri = "ws://127.0.0.1:1/showdown/websocket"

            async def start(self) -> None:
                return None

            async def close(self) -> None:
                return None

        class FakeBridge:
            def __init__(self, **_: object) -> None:
                return None

            async def start(self) -> None:
                return None

            async def close(self) -> None:
                return None

        calls = 0

        async def wait_for_challenge(**_: object) -> None:
            return None

        async def run_single_game(**_: object) -> ControlledFoulPlayGameResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ControlledFoulPlayGameResult(
                    battle_id="battle-1",
                    seed=1,
                    winner="PokeZeroBot",
                    pokezero_won=True,
                    decision_rounds=1,
                    pokezero_decisions=1,
                    root_puct_searches=1,
                    root_puct_fallbacks=0,
                )
            raise RuntimeError("simulated game failure")

        async def spawn_foulplay(*_: object, **__: object) -> FakeProcess:
            return FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "nested" / "summary.json"
            argv = (
                "--checkpoint",
                "checkpoint.pt",
                "--showdown-root",
                "showdown",
                "--games",
                "2",
                "--summary-out",
                str(summary_path),
            )
            with (
                patch("pokezero.foulplay_bridge._validate_external_paths"),
                patch(
                    "pokezero.foulplay_bridge.load_transformer_checkpoint",
                    return_value=(object(), FakeCheckpointResult()),
                ),
                patch("pokezero.foulplay_bridge.category_vocab_from_model_config", return_value=object()),
                patch("pokezero.foulplay_bridge.load_showdown_dex_cached", return_value=object()),
                patch("pokezero.foulplay_bridge._build_policy", return_value=FakePolicy()),
                patch("pokezero.foulplay_bridge._FoulPlayWebsocketServer", FakeServer),
                patch("pokezero.foulplay_bridge._BattleBridge", FakeBridge),
                patch("pokezero.foulplay_bridge._spawn_foulplay", side_effect=spawn_foulplay),
                patch("pokezero.foulplay_bridge._wait_for_foulplay_challenge_or_exit", side_effect=wait_for_challenge),
                patch("pokezero.foulplay_bridge._run_single_game", side_effect=run_single_game),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated game failure"):
                    asyncio.run(async_main(argv))

            payload = json.loads(summary_path.read_text())

        self.assertEqual(payload["completed_games"], 1)
        self.assertEqual(payload["games"], 2)
        self.assertEqual(payload["wins"], 1)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["complete"], False)


class HeadToHeadOpponentTest(unittest.TestCase):
    """The opponent seat can be a second pokezero policy instead of foul-play.

    Why this mode exists: measuring what search buys over the raw model THROUGH foul-play
    is underpowered. The same control config scored 0.581 then 0.596 on identical seeds --
    a 1.5-2.1pp noise floor -- while the effect being chased was +2.95pp at p=0.56. A
    direct head-to-head spends every game on the difference instead of diluting it through
    a third party.
    """

    def _cfg(self, **kw):
        from pokezero.foulplay_bridge import ControlledFoulPlayConfig
        base = dict(checkpoint=Path("/tmp/ckpt.pt"), showdown_root=Path("/tmp/showdown"))
        base.update(kw)
        return ControlledFoulPlayConfig(**base)

    def test_default_is_unchanged_foul_play(self) -> None:
        # The whole existing campaign history depends on this default. A silent change
        # would re-point every banked comparison at a different opponent.
        self.assertEqual(self._cfg(policy_mode="raw").opponent_policy_mode, "foul-play")

    def test_a_mirror_match_is_refused(self) -> None:
        """Identical policies on both seats score 0.5 by construction.

        Refused rather than run: it would produce a plausible 0.5 that reads as "search
        does not help" when in fact nothing was compared.
        """
        with self.assertRaises(ValueError) as caught:
            self._cfg(policy_mode="raw", opponent_policy_mode="raw")
        self.assertIn("mirror match", str(caught.exception))

    def test_an_unknown_opponent_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._cfg(policy_mode="raw", opponent_policy_mode="totally-bogus")

    def test_the_summary_names_the_opponent_that_actually_played(self) -> None:
        """Three summary sites hardcoded "foul-play".

        Downstream analyzers key arms off `opponent_policy_id`, so leaving the hardcode in
        place would pool a head-to-head arm with a foul-play arm as though they shared an
        opponent.
        """
        from pokezero.foulplay_bridge import _opponent_policy_id_label
        self.assertEqual(
            _opponent_policy_id_label(self._cfg(policy_mode="raw")), "foul-play")
        self.assertEqual(
            _opponent_policy_id_label(
                self._cfg(policy_mode="engine-mcts", opponent_policy_mode="raw",
                          engine_model_path=Path("/tmp/m.pt"),
                          engine_tables_path=Path("/tmp/t.json"))),
            "pokezero-raw")

    def test_the_opponent_display_name_follows_who_actually_plays(self) -> None:
        """A head-to-head shard must not record "FoulPlayBot" as the winner.

        Observed on the first 16k probe: the raw model won and the summary said
        `winner: "FoulPlayBot"`. `_winner_name` and `_terminal_from_public_lines` both key
        off this string, so the battle log, the title line and the winner field would all
        name an opponent that never played a move. An explicit --foulplay-username still
        wins, so this cannot silently rename a deliberately-labelled run.
        """
        engine = dict(policy_mode="engine-mcts",
                      engine_model_path=Path("/tmp/m.pt"),
                      engine_tables_path=Path("/tmp/t.json"))
        self.assertEqual(self._cfg(**engine).foulplay_username, "FoulPlayBot")
        self.assertEqual(
            self._cfg(**engine, opponent_policy_mode="raw").foulplay_username,
            "PokeZeroRawBot")
        self.assertEqual(
            self._cfg(**engine, opponent_policy_mode="raw",
                      foulplay_username="Explicit").foulplay_username,
            "Explicit")

    def test_budget_versus_budget_is_expressible_and_not_a_mirror(self) -> None:
        """d3/s2048 against d6/s16384 on one board.

        Without per-seat axes the opponent inherits depth and sims, so engine-vs-engine is
        necessarily a mirror and the only head-to-head expressible is search-vs-raw. This
        is the comparison that decides whether a cheap configuration can replace an
        expensive one, and it is far more sensitive than scoring each against a third
        policy and differencing.
        """
        from pokezero.foulplay_bridge import _opponent_seat_config
        engine = dict(engine_model_path=Path("/tmp/m.pt"), engine_tables_path=Path("/tmp/t.json"))
        cfg = self._cfg(policy_mode="engine-mcts", engine_depth=3, engine_sims=2048,
                        opponent_policy_mode="engine-mcts",
                        opponent_engine_depth=6, opponent_engine_sims=16384, **engine)
        opp = _opponent_seat_config(cfg)
        self.assertEqual((cfg.engine_depth, cfg.engine_sims), (3, 2048))
        self.assertEqual((opp.engine_depth, opp.engine_sims), (6, 16384))
        self.assertEqual(opp.policy_mode, "engine-mcts")
        # The derived config builds ONE policy, so the pairing fields must be cleared --
        # otherwise an opponent axis survives with opponent_policy_mode reset to
        # foul-play, which the axis guard rejects. (It did, before this was fixed.)
        self.assertIsNone(opp.opponent_engine_depth)
        self.assertIsNone(opp.opponent_engine_sims)
        self.assertEqual(opp.opponent_policy_mode, "foul-play")

    def test_same_mode_and_same_axes_is_still_refused_as_a_mirror(self) -> None:
        engine = dict(engine_model_path=Path("/tmp/m.pt"), engine_tables_path=Path("/tmp/t.json"))
        with self.assertRaises(ValueError) as caught:
            self._cfg(policy_mode="engine-mcts", engine_depth=6, engine_sims=16384,
                      opponent_policy_mode="engine-mcts", **engine)
        self.assertIn("mirror match", str(caught.exception))

    def test_an_opponent_axis_on_a_non_engine_opponent_is_refused(self) -> None:
        """Refused, not ignored: the shard's config echo is the only record of WHICH two
        things were compared, so an opponent depth the opponent never used would
        misdescribe the pairing."""
        engine = dict(engine_model_path=Path("/tmp/m.pt"), engine_tables_path=Path("/tmp/t.json"))
        with self.assertRaises(ValueError):
            self._cfg(policy_mode="engine-mcts", opponent_policy_mode="raw",
                      opponent_engine_depth=6, **engine)

    def test_oracle_belief_with_an_engine_opponent_is_refused(self) -> None:
        """The oracle override is installed for the pokezero seat only.

        _install_oracle_belief_override runs inside the pokezero-seat branch, so an
        engine-mcts opponent would search SAMPLED beliefs while the shard echoes
        oracle_belief: true for the whole cell. That is a false witness on the only record,
        and the run would silently be oracle-vs-sampled on top of whatever it meant to
        compare. Refused until the override is installed for both seats.
        """
        engine = dict(engine_model_path=Path("/tmp/m.pt"), engine_tables_path=Path("/tmp/t.json"))
        with self.assertRaises(ValueError) as caught:
            self._cfg(policy_mode="engine-mcts", engine_oracle_belief=True,
                      opponent_policy_mode="engine-mcts", opponent_engine_depth=6, **engine)
        self.assertIn("oracle", str(caught.exception).lower())
        # Still allowed where the override does apply, and where there is no engine opponent.
        self._cfg(policy_mode="engine-mcts", engine_oracle_belief=True, **engine)
        self._cfg(policy_mode="engine-mcts", opponent_policy_mode="raw", **engine)

    def test_the_opponent_seats_search_health_is_recorded_separately(self) -> None:
        """A budget comparison whose opponent silently fell back would read as a tie.

        Every engine health aggregate is derived from state.decisions, which is gated on
        the pokezero seat, so a d6 opponent falling back on most decisions would leave the
        shard reporting fallback_rate 0.0 -- a figure describing only the OTHER seat -- and
        the campaign would conclude the cheap config matches the expensive one. This is the
        already-observed failure shape: a None public state once made engine-MCTS play
        uniform-legal while reporting no error (0/20 against raw's 10/20).
        """
        from pokezero.foulplay_bridge import ControlledFoulPlayGameResult, _ControlledBattleState
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ControlledFoulPlayGameResult)}
        for name in ("opponent_engine_mcts_decisions", "opponent_engine_mcts_fallbacks",
                     "opponent_engine_mcts_fallback_reasons"):
            self.assertIn(name, fields, f"{name} must reach the game result")
        state_fields = {f.name for f in dataclasses.fields(_ControlledBattleState)}
        self.assertIn("opponent_decisions", state_fields)
        # The pokezero-seat aggregate keeps its own name, so existing readers are unchanged.
        self.assertIn("engine_mcts_decisions", fields)

    def test_room_lines_are_a_no_op_only_when_explicitly_allowed(self) -> None:
        """With no foul-play client, a room-line send must not abort the battle.

        Every room-line call site is "tell the foul-play client what happened" -- the init
        line, the private-line forwarding, the terminal notify. In head-to-head there is
        nobody to tell, and raising would kill each battle at its first line. But the
        default MUST still raise, or a genuinely dropped foul-play connection becomes
        silence instead of a fault.
        """
        from pokezero.foulplay_bridge import _FoulPlayWebsocketServer, FoulPlayProtocolError
        strict = _FoulPlayWebsocketServer(username="fp", host="127.0.0.1")
        with self.assertRaises(FoulPlayProtocolError):
            asyncio.run(strict.send_room_lines("battle-1", ["|init|battle"]))
        lenient = _FoulPlayWebsocketServer(
            username="fp", host="127.0.0.1", allow_missing_client=True)
        asyncio.run(lenient.send_room_lines("battle-1", ["|init|battle"]))  # must not raise

    def test_both_seats_are_packed_by_the_same_code(self) -> None:
        """_context_for_seat must be seat-agnostic.

        If the opponent's observation were packed differently from the pokezero seat's, a
        measured strength difference could be the packing rather than the search -- the
        exact confound this mode exists to avoid. Asserted by building a context for each
        seat from one observations dict and checking the seat-dependent fields track the
        seat argument.
        """
        from pokezero.foulplay_bridge import _context_for_seat

        class _Obs:
            legal_action_mask = (True,) * ACTION_COUNT
            schema_version = 1
            metadata: dict = {}

        observations = {"p1": _Obs(), "p2": _Obs()}
        state = SimpleNamespace(battle_id="battle-x", seed=7, trajectory=None)
        cfg = self._cfg(policy_mode="raw")
        contexts = {
            seat: _context_for_seat(
                seat=seat,
                policy=SimpleNamespace(),          # no public materialization needed
                state=state,
                config=cfg,
                observations=observations,
                requested_players=("p1", "p2"),
                decision_round=3,
                belief_set_source="public",
            )
            for seat in ("p1", "p2")
        }
        self.assertEqual(contexts["p1"].player_id, "p1")
        self.assertEqual(contexts["p2"].player_id, "p2")
        for seat, ctx in contexts.items():
            with self.subTest(seat=seat):
                self.assertEqual(ctx.decision_round_index, 3)
                self.assertEqual(ctx.battle_id, "battle-x")
                self.assertEqual(ctx.requested_players, ("p1", "p2"))
                # the acting seat is always present in its own mask set
                self.assertIn(seat, ctx.requested_legal_action_masks)


# One foul-play decision's stdout, VERBATIM in shape: the messages are the ones
# `fp/search/main.py` logs at the pinned submodule commit (9955255), rendered through
# `config.py`'s `CustomFormatter`, which writes `levelname.ljust(8) + " " + msg`.
_FOULPLAY_DECISION_STDOUT = (
    "INFO     Searching for a move using MCTS...",
    "INFO     Sampling 2 battles at 1000ms each",
    "DEBUG    Calling with 0 state: gyarados,100,...",
    "INFO     Iterations 0: 41234",
    "INFO     Iterations 1: 39871",
    "INFO     Policy 0: icebeam visited 61.0% avg_score=0.512 sample_chance_multiplier=0.5",
    "INFO     Considered Choices:",
    "INFO     \t61.0%: icebeam",
    "INFO     Choice: icebeam",
)
_FOULPLAY_REQUEST_LINE = '|request|{"active":[{"moves":[{"id":"icebeam"}]}],"side":{"id":"p1"},"rqid":3}'


class _ThinkBridge:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, payload: dict) -> None:
        self.messages.append(payload)


class _ThinkPlayerState:
    def __init__(self, slot: str) -> None:
        self.slot = slot


def _think_observation(slot: str) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=tuple(index == 0 for index in range(ACTION_COUNT)),
        metadata={"slot": slot},
    )


@contextlib.contextmanager
def _stubbed_seat_packing():
    """Stub the seat packing only -- the wait, the clock and the parser stay REAL.

    Every other decision-boundary test in this file patches
    `_wait_for_foulplay_choice_or_exit` out, which is exactly the case where the clock
    goes unstamped. These tests instead drive the real wait over a fake websocket and a
    fake process, so a measured wait is paired with a parsed iteration count through the
    shipping code path.
    """

    with (
        patch(
            "pokezero.foulplay_bridge._player_state",
            side_effect=lambda _state, slot, **_kwargs: _ThinkPlayerState(slot),
        ),
        patch(
            "pokezero.foulplay_bridge.observation_from_player_state",
            side_effect=lambda player_state, **_kwargs: _think_observation(player_state.slot),
        ),
        patch(
            "pokezero.foulplay_bridge._observation_with_search_metadata",
            side_effect=lambda value, _state: value,
        ),
        patch(
            "pokezero.foulplay_bridge._select_policy_decision",
            side_effect=lambda *_args, **_kwargs: PolicyDecision(action_index=0, policy_id="stub"),
        ),
        patch(
            "pokezero.foulplay_bridge.showdown_choice_for_action",
            side_effect=lambda player_state, action: f"{player_state.slot}:{action}",
        ),
        patch(
            "pokezero.foulplay_bridge.action_index_from_choice_string",
            side_effect=lambda _state, _choice: 0,
        ),
    ):
        yield


class _ThinkProcess:
    """A foul-play process that is alive and never exits on its own."""

    returncode: int | None = None

    async def wait(self) -> int:
        await asyncio.sleep(3600)
        return 0


class _ThinkServer:
    """foul-play's websocket and its stdout, and nothing else about it.

    `think_seconds` is how long the choice takes to come back; `stdout` is what the
    process printed while doing it. `late_seconds` publishes the per-sample lines AFTER
    the choice has already been returned, which is the real drain race: the choice
    arrives on the websocket while the log lines are still in the pipe.
    """

    def __init__(
        self,
        logs: _ProcessLogBuffer,
        *,
        stdout: tuple[str, ...] = _FOULPLAY_DECISION_STDOUT,
        think_seconds: float = 0.03,
        late_seconds: float | None = None,
    ) -> None:
        self.logs = logs
        self.stdout = stdout
        self.think_seconds = think_seconds
        self.late_seconds = late_seconds
        self.sent: list[tuple[str, tuple[str, ...]]] = []

    async def send_room_lines(self, battle_id: str, lines) -> None:
        self.sent.append((battle_id, tuple(lines)))

    async def wait_for_choice(self, *, battle_id: str) -> str:
        announcements = [line for line in self.stdout if "Iterations" not in line]
        results = [line for line in self.stdout if "Iterations" in line]
        for line in announcements:
            self.logs.append_stdout(line)
        await asyncio.sleep(self.think_seconds)
        if self.late_seconds is None:
            for line in results:
                self.logs.append_stdout(line)
        else:
            loop = asyncio.get_running_loop()

            def publish() -> None:
                for line in results:
                    self.logs.append_stdout(line)

            loop.call_later(self.late_seconds, publish)
        return "move icebeam"


# The per-decision delimiter `find_best_move` logs from the PARENT process
# (`fp/search/main.py:129`), so it survives every start method. Every well-formed slice
# carries exactly one.
_FOULPLAY_MARKER = "INFO     Searching for a move using MCTS..."


def _decision_slice(*lines: str) -> tuple[str, ...]:
    """One decision's worth of instrument lines, delimiter included."""

    return (_FOULPLAY_MARKER, *lines)


class FoulPlayThinkParserTest(unittest.TestCase):
    """The realized-work parser: `fp/search/main.py:57`, `:129` and `:130-132`."""

    def test_parses_a_real_decision_and_reports_realized_work(self) -> None:
        work = _foulplay_think_work_from_log_lines(_FOULPLAY_DECISION_STDOUT)

        self.assertEqual(work.miss_reasons, ())
        self.assertEqual(work.iterations_per_sample, (41234, 39871))
        self.assertEqual(work.declared_samples, 2)
        self.assertEqual(work.budget_ms_per_sample, 1000)
        self.assertEqual(work.decision_markers, 1)
        self.assertEqual(work.total_iterations, 81105)
        self.assertEqual(work.budget_seconds_granted, 2.0)
        self.assertAlmostEqual(work.iterations_per_budget_second, 81105 / 2.0)

    def test_the_visit_count_is_read_after_the_colon_not_before_it(self) -> None:
        """`Iterations {index}: {total_visits}` -- index FIRST, realized work SECOND.

        A parser keyed on the first integer in the line records 0 and 1, the sampled
        battle indices, which look like a plausible iteration count and measure nothing.
        """

        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 2 battles at 1000ms each",
                "INFO     Iterations 0: 41234",
                "INFO     Iterations 1: 39871",
            )
        )

        self.assertEqual(work.iterations_per_sample, (41234, 39871))
        self.assertNotIn(0, work.iterations_per_sample)
        self.assertNotIn(1, work.iterations_per_sample)

    def test_a_genuine_zero_is_reported_as_a_parsed_zero(self) -> None:
        """The rule is "no zero that MEANS unparsed", not "no zero"."""

        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 1 battles at 1000ms each", "INFO     Iterations 0: 0"
            )
        )

        self.assertEqual(work.miss_reasons, ())
        self.assertEqual(work.total_iterations, 0)
        self.assertEqual(work.iterations_per_budget_second, 0.0)

    def test_a_malformed_iterations_line_is_a_miss_not_a_zero(self) -> None:
        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 2 battles at 1000ms each",
                "INFO     Iterations 3:",
                "INFO     Iterations 4: many",
            )
        )

        self.assertIn("iterations_line_malformed", work.miss_reasons)
        self.assertIsNone(work.total_iterations)
        self.assertIsNone(work.iterations_per_budget_second)

    def test_an_unrelated_line_mentioning_iterations_does_not_refuse_the_decision(self) -> None:
        r"""A false MISS is safe but not free: it also re-arms the settle give-up.

        The trigger is `\bIterations\s+\d`, not the bare word, so an unrelated DEBUG
        line cannot cost a decision its measurement.
        """

        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 2 battles at 1000ms each",
                "DEBUG    Total Iterations this turn were fine",
                "DEBUG    MyIterations 0: 999",
                "INFO     Iterations 0: 41234",
                "INFO     Iterations 1: 39871",
            )
        )

        self.assertEqual(work.miss_reasons, ())
        self.assertEqual(work.total_iterations, 81105)

    def test_an_implausible_visit_count_is_a_miss(self) -> None:
        """A corrupted line parses as a perfectly good integer and poisons a mean."""

        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 1 battles at 1000ms each",
                "INFO     Iterations 0: 9999999999999999999999999999999999999999",
            )
        )

        self.assertIn("iterations_implausible", work.miss_reasons)
        self.assertIsNone(work.total_iterations)
        self.assertIsNone(work.iterations_per_budget_second)

    def test_a_missing_sampling_line_refuses_the_total(self) -> None:
        work = _foulplay_think_work_from_log_lines(
            _decision_slice("INFO     Iterations 0: 41234", "INFO     Iterations 1: 39871")
        )

        self.assertEqual(work.miss_reasons, ("sampling_line_absent",))
        self.assertIsNone(work.total_iterations)
        self.assertIsNone(work.budget_seconds_granted)

    def test_fewer_iterations_lines_than_declared_refuses_the_total(self) -> None:
        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 2 battles at 1000ms each", "INFO     Iterations 0: 41234"
            )
        )

        self.assertIn("sample_count_mismatch", work.miss_reasons)
        self.assertIsNone(work.total_iterations)

    def test_more_iterations_lines_than_declared_refuses_the_total(self) -> None:
        """A late line landing in the NEXT decision's slice must not inflate its total."""

        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 2 battles at 1000ms each",
                "INFO     Iterations 0: 41234",
                "INFO     Iterations 1: 39871",
                "INFO     Iterations 1: 39871",
            )
        )

        self.assertIn("sample_count_mismatch", work.miss_reasons)
        self.assertIsNone(work.total_iterations)

    def test_two_sampling_announcements_in_one_slice_refuse_the_total(self) -> None:
        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 1 battles at 1000ms each",
                "INFO     Iterations 0: 41234",
                "INFO     Sampling 1 battles at 1000ms each",
                "INFO     Iterations 0: 39871",
            )
        )

        self.assertIn("sampling_line_repeated", work.miss_reasons)
        self.assertIsNone(work.total_iterations)

    def test_a_slice_with_no_decision_marker_is_not_attributable(self) -> None:
        """Round N's output landing in round N+1's slice: internally perfect, wrong round.

        Without the delimiter this reads as a flawless `ok` record -- the exact shape that
        pairs one decision's realized work with another decision's wait.
        """

        work = _foulplay_think_work_from_log_lines(
            (
                "INFO     Sampling 2 battles at 1000ms each",
                "INFO     Iterations 0: 41234",
                "INFO     Iterations 1: 39871",
            )
        )

        self.assertEqual(work.decision_markers, 0)
        self.assertIn("decision_marker_absent", work.miss_reasons)
        self.assertIsNone(work.total_iterations)

    def test_two_decision_markers_in_one_slice_refuse_the_total(self) -> None:
        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 1 battles at 1000ms each",
                "INFO     Iterations 0: 41234",
                _FOULPLAY_MARKER,
            )
        )

        self.assertEqual(work.decision_markers, 2)
        self.assertIn("decision_marker_repeated", work.miss_reasons)
        self.assertIsNone(work.total_iterations)

    def test_an_empty_slice_is_absence_not_zero_work(self) -> None:
        work = _foulplay_think_work_from_log_lines(())

        self.assertEqual(
            set(work.miss_reasons),
            {"iterations_line_absent", "sampling_line_absent", "decision_marker_absent"},
        )
        self.assertIsNone(work.total_iterations)

    def test_a_nonpositive_budget_refuses_the_rate(self) -> None:
        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 2 battles at 0ms each",
                "INFO     Iterations 0: 41234",
                "INFO     Iterations 1: 39871",
            )
        )

        self.assertIn("budget_not_positive", work.miss_reasons)
        self.assertIsNone(work.budget_seconds_granted)
        self.assertIsNone(work.iterations_per_budget_second)

    def test_an_unrecognised_sampling_line_is_a_miss(self) -> None:
        """If foul-play's format changes, the instrument REFUSES rather than guesses."""

        work = _foulplay_think_work_from_log_lines(
            _decision_slice(
                "INFO     Sampling 2 battles at 1000 milliseconds each",
                "INFO     Iterations 0: 41234",
            )
        )

        self.assertIn("sampling_line_malformed", work.miss_reasons)
        self.assertIsNone(work.total_iterations)


class ProcessLogBufferSliceTest(unittest.TestCase):
    """The instrument's own cursor buffer, separate from the crash ring."""

    def test_only_instrument_lines_are_kept_so_debug_cannot_displace_them(self) -> None:
        """foul-play logs at DEBUG. Chatter must not push a measurement out of reach.

        This is the B7 fix: with the measurement read off the 200-line crash ring, 263
        DEBUG lines in one decision made every row `log_lines_dropped` -- a total,
        silent, favourable-direction failure.
        """

        logs = _ProcessLogBuffer()
        logs.append_stdout(_FOULPLAY_MARKER)
        logs.append_stdout("INFO     Sampling 2 battles at 1000ms each")
        for index in range(2000):
            logs.append_stdout(f"DEBUG    Calling with {index} state: gyarados,100,...")
        logs.append_stdout("INFO     Iterations 0: 41234")
        logs.append_stdout("INFO     Iterations 1: 39871")

        log_slice = logs.think_lines_since(0)
        work = _foulplay_think_work_from_log_lines(log_slice.lines, dropped=log_slice.dropped)

        self.assertEqual(log_slice.dropped, 0)
        self.assertEqual(len(log_slice.lines), 4)
        self.assertEqual(work.miss_reasons, ())
        self.assertEqual(work.total_iterations, 81105)
        # The crash ring still behaves as before, and still holds only its 200-line tail.
        self.assertEqual(len(logs.stdout), 200)

    def test_slice_returns_only_new_lines_and_advances_the_cursor(self) -> None:
        logs = _ProcessLogBuffer()
        for line in (_FOULPLAY_MARKER, "INFO     Iterations 0: 1"):
            logs.append_stdout(line)

        first = logs.think_lines_since(0)
        logs.append_stdout("INFO     Iterations 1: 2")
        second = logs.think_lines_since(first.cursor)

        self.assertEqual(first.lines, (_FOULPLAY_MARKER, "INFO     Iterations 0: 1"))
        self.assertEqual(first.dropped, 0)
        self.assertEqual(second.lines, ("INFO     Iterations 1: 2",))
        self.assertEqual(second.cursor, 3)

    def test_the_bound_reports_what_it_discarded_and_the_parse_refuses(self) -> None:
        """Overflowing the instrument buffer itself must refuse, not sum what survived.

        Reaching this now takes `_FOULPLAY_THINK_LINE_CAP` INSTRUMENT lines (~50 decisions
        of backlog), not 200 lines of DEBUG chatter -- but if it happens the surviving
        `Iterations` lines would still sum to a plausible total missing an unknown number
        of samples.
        """

        logs = _ProcessLogBuffer()
        logs.append_stdout(_FOULPLAY_MARKER)
        logs.append_stdout("INFO     Sampling 2 battles at 1000ms each")
        for index in range(_FOULPLAY_THINK_LINE_CAP * 2):
            logs.append_stdout(f"INFO     Iterations {index}: 41234")

        log_slice = logs.think_lines_since(0)
        work = _foulplay_think_work_from_log_lines(log_slice.lines, dropped=log_slice.dropped)

        self.assertGreater(log_slice.dropped, 0)
        self.assertEqual(len(log_slice.lines), _FOULPLAY_THINK_LINE_CAP)
        self.assertIn("log_lines_dropped", work.miss_reasons)
        self.assertIsNone(work.total_iterations)

    def test_a_cursor_past_the_end_yields_nothing_rather_than_raising(self) -> None:
        logs = _ProcessLogBuffer()
        logs.append_stdout(_FOULPLAY_MARKER)

        log_slice = logs.think_lines_since(99)

        self.assertEqual(log_slice.lines, ())
        self.assertEqual(log_slice.dropped, 0)
        self.assertEqual(log_slice.cursor, 1)

    def test_a_negative_cursor_does_not_fabricate_drops(self) -> None:
        """The docstring promised no drops on a bad cursor; a negative one broke it."""

        logs = _ProcessLogBuffer()
        for index in range(5):
            logs.append_stdout(f"INFO     Iterations {index}: 1")

        log_slice = logs.think_lines_since(-3)

        self.assertEqual(log_slice.dropped, 0)
        self.assertEqual(len(log_slice.lines), 5)

class FoulPlayThinkClockTest(unittest.TestCase):
    def test_the_wait_stamps_the_clock_with_the_time_the_choice_took(self) -> None:
        logs = _ProcessLogBuffer()
        server = _ThinkServer(logs, think_seconds=0.05)
        clock = _FoulPlayThinkClock()

        choice = asyncio.run(
            _wait_for_foulplay_choice_or_exit(
                server=server,  # type: ignore[arg-type]
                battle_id="battle-1",
                process=_ThinkProcess(),  # type: ignore[arg-type]
                logs=logs,
                clock=clock,
            )
        )

        self.assertEqual(choice, "move icebeam")
        self.assertIsNotNone(clock.wait_seconds)
        self.assertGreaterEqual(clock.wait_seconds, 0.05)

    def test_an_exited_process_leaves_the_clock_unstamped(self) -> None:
        """The wait raised, so there is no wait to report -- and None, not 0.0."""

        logs = _ProcessLogBuffer()
        clock = _FoulPlayThinkClock()
        process = _ThinkProcess()
        process.returncode = 137

        with self.assertRaises(FoulPlayProcessExitError):
            asyncio.run(
                _wait_for_foulplay_choice_or_exit(
                    server=_ThinkServer(logs),  # type: ignore[arg-type]
                    battle_id="battle-1",
                    process=process,  # type: ignore[arg-type]
                    logs=logs,
                    clock=clock,
                )
            )

        self.assertIsNotNone(clock.started_at)
        self.assertIsNone(clock.finished_at)
        self.assertIsNone(clock.wait_seconds)


class _ThinkBoundaryDriver:
    """Shared driver for the boundary tests.

    A plain mixin rather than a base TestCase: subclassing a TestCase to reuse its helpers
    re-runs every one of its tests once per subclass, which inflates the counts this PR
    reports and hides which class actually exercised what.
    """

    def _state(self) -> _ControlledBattleState:
        return _ControlledBattleState(
            battle_id="battle-7",
            seed=7,
            format_id="gen3randombattle",
            trajectory=BattleTrajectory(battle_id="battle-7", format_id="gen3randombattle", seed=7),
        )

    def _config(self, **overrides) -> ControlledFoulPlayConfig:
        return ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            pokezero_player="p2",
            **overrides,
        )

    def _run(
        self,
        *,
        state: _ControlledBattleState,
        config: ControlledFoulPlayConfig,
        server,
        logs,
        rounds: int = 1,
        forward_request_before_round: tuple[int, ...] = (0,),
        our_search_seconds: float = 0.02,
    ) -> _ThinkBridge:
        bridge = _ThinkBridge()

        async def scenario() -> None:
            for decision_round in range(rounds):
                if decision_round in forward_request_before_round:
                    # The REAL forwarding path, which is where the opponent's clock
                    # starts: this happens before the decision boundary runs.
                    await _handle_stream_event(
                        state,
                        server,
                        {"stream": "p1", "lines": [_FOULPLAY_REQUEST_LINE]},
                        config=config,
                    )
                # Stands in for our own search, which runs while foul-play already holds
                # the request.
                await asyncio.sleep(our_search_seconds)
                await _handle_decision_boundary(
                    config=config,
                    bridge=bridge,  # type: ignore[arg-type]
                    server=server,
                    state=state,
                    policy=object(),
                    vocab=object(),
                    dex=object(),
                    observation_spec=SimpleNamespace(schema_version="v2.2"),
                    decision_round=decision_round,
                    requested_players=("p1", "p2"),
                    foulplay_process=_ThinkProcess(),
                    foulplay_logs=logs,
                )

        with _stubbed_seat_packing():
            asyncio.run(scenario())
        return bridge


class FoulPlayThinkBoundaryTest(_ThinkBoundaryDriver, unittest.TestCase):
    """The instrument as a whole, over the real wait and the real parser."""

    def test_a_decision_records_its_wait_and_the_work_foul_play_realized(self) -> None:
        logs = _ProcessLogBuffer()
        state = self._state()
        config = self._config()

        bridge = self._run(
            state=state,
            config=config,
            server=_ThinkServer(logs, think_seconds=0.05),
            logs=logs,
            our_search_seconds=0.03,
        )

        self.assertEqual(len(state.opponent_think), 1)
        row = state.opponent_think[0]
        self.assertEqual(row["status"], "ok")
        self.assertNotIn("miss_reasons", row)
        self.assertEqual(row["round"], 0)
        self.assertEqual(row["iterations"], 81105)
        self.assertEqual(row["iterations_per_sample"], [41234, 39871])
        self.assertEqual(row["sampled_battles"], 2)
        # Absent BECAUSE it matched `sampled_battles`; present only on a mismatch.
        self.assertNotIn("declared_sampled_battles", row)
        self.assertEqual(row["budget_ms_per_sample"], 1000)
        self.assertAlmostEqual(row["iterations_per_budget_second"], 81105 / 2.0)
        # The schedule this decision ran under, DERIVED from the parsed announcement rather
        # than from the run config, because foul-play recomputes it every decision.
        self.assertEqual(row["stratum"], "2x1000ms")
        # The wait STARTS after our search finished, so it sees only foul-play's tail...
        self.assertGreaterEqual(row["wait_seconds"], 0.05)
        # ...and the overlap window is our own search, running while foul-play already
        # had the request. That is the CPU-contention window this arm has to defend.
        self.assertGreaterEqual(row["overlap_seconds"], 0.03)
        self.assertEqual(state.opponent_think_failures, 0)
        # The same block per decision, on the decision and in the trajectory.
        decision_metadata = state.trajectory.steps[0].metadata
        self.assertEqual(decision_metadata["policy_id"], "foul-play")
        self.assertEqual(decision_metadata["foulplay_think"]["iterations"], 81105)
        self.assertEqual(bridge.messages[0]["choices"]["p1"], "move icebeam")

    def test_lines_still_in_the_pipe_are_waited_out_within_the_settle(self) -> None:
        """The choice arrives on the websocket before the stdout drain catches up."""

        logs = _ProcessLogBuffer()
        state = self._state()

        self._run(
            state=state,
            config=self._config(),
            server=_ThinkServer(logs, think_seconds=0.0, late_seconds=0.005),
            logs=logs,
        )

        row = state.opponent_think[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["iterations"], 81105)
        # Emitted BECAUSE the race bit: the lines were still in the pipe.
        self.assertGreater(row["settle_seconds"], 0.0)

    def test_an_opponent_that_never_logs_records_a_miss_never_zero_iterations(self) -> None:
        """The `spawn` case: foul-play's pool child has no log handler at all.

        Under `spawn` (macOS, or Linux with a forkserver default) the `Iterations` line is
        never emitted. The instrument must say so on every decision rather than reporting
        an opponent that did no work -- and must stop paying the settle for lines that are
        not coming.
        """

        logs = _ProcessLogBuffer()
        state = self._state()
        rounds = _FOULPLAY_THINK_SETTLE_GIVEUP_DECISIONS + 1

        self._run(
            state=state,
            config=self._config(),
            server=_ThinkServer(logs, stdout=(), think_seconds=0.0),
            logs=logs,
            rounds=rounds,
            forward_request_before_round=tuple(range(rounds)),
            our_search_seconds=0.0,
        )

        self.assertEqual(len(state.opponent_think), rounds)
        for row in state.opponent_think:
            with self.subTest(round=row["round"]):
                self.assertEqual(row["status"], "miss")
                self.assertIn("iterations_line_absent", row["miss_reasons"])
                self.assertNotIn("iterations", row)
                self.assertNotIn("iterations_per_budget_second", row)
                # On a miss the row describes the evidence without PRICING it: publishing
                # `iterations_per_sample` + `sampled_battles` + `budget_ms_per_sample` would
                # let a consumer recompute the very rate the guard withheld.
                self.assertNotIn("iterations_per_sample", row)
                self.assertNotIn("sampled_battles", row)
                self.assertNotIn("budget_ms_per_sample", row)
                self.assertEqual(row["iterations_lines_seen"], 0)
                # The wait itself is still measured: the opponent answered, we just
                # cannot say what it accomplished.
                self.assertIn("wait_seconds", row)
        self.assertEqual(state.foulplay_think_unusable_streak, rounds)
        # The settle was given up on: after the streak, no measurable wait was spent on
        # lines that are not coming, so the field is not even emitted.
        self.assertNotIn("settle_seconds", state.opponent_think[-1])
        self.assertLess(
            state.opponent_think[-1].get("settle_seconds", 0.0),
            _FOULPLAY_THINK_SETTLE_SECONDS,
        )

    def test_a_decision_with_no_forwarded_request_refuses_the_overlap_window(self) -> None:
        """Round 1 gets no request of its own, so round 0's stamp must not be reused."""

        logs = _ProcessLogBuffer()
        state = self._state()

        self._run(
            state=state,
            config=self._config(),
            server=_ThinkServer(logs, think_seconds=0.0),
            logs=logs,
            rounds=2,
            forward_request_before_round=(0,),
            our_search_seconds=0.0,
        )

        first, second = state.opponent_think
        self.assertIn("overlap_seconds", first)
        self.assertNotIn("overlap_seconds", second)
        self.assertEqual(second["status"], "miss")
        self.assertIn("request_forward_unmeasured", second["miss_reasons"])

    def test_a_stubbed_wait_records_wait_unmeasured_rather_than_zero_seconds(self) -> None:
        """What every other boundary test in this file does, read as a measurement.

        A patched or replaced wait leaves the clock unstamped. `0.0` would be a lie that
        drags a run's mean wait toward zero; the row says the wait was not measured.
        """

        logs = _ProcessLogBuffer()
        state = self._state()
        for line in _FOULPLAY_DECISION_STDOUT:
            logs.append_stdout(line)

        async def stubbed_wait(**_kwargs) -> str:
            return "move icebeam"

        with patch(
            "pokezero.foulplay_bridge._wait_for_foulplay_choice_or_exit",
            side_effect=stubbed_wait,
        ):
            self._run(
                state=state,
                config=self._config(),
                server=_ThinkServer(logs),
                logs=logs,
                our_search_seconds=0.0,
            )

        row = state.opponent_think[0]
        self.assertEqual(row["status"], "miss")
        self.assertIn("wait_unmeasured", row["miss_reasons"])
        self.assertNotIn("wait_seconds", row)
        self.assertNotIn("overlap_seconds", row)
        # The realized work still parses: the two halves fail independently.
        self.assertEqual(row["iterations"], 81105)

    def test_telemetry_that_raises_is_counted_and_the_battle_continues(self) -> None:
        """The isolation counter, read on an input that makes it fire."""

        class BrokenLogs:
            def stdout_since(self, _cursor):
                raise RuntimeError("log buffer exploded")

            def tail(self) -> str:
                return ""

        state = self._state()
        logs = BrokenLogs()

        bridge = self._run(
            state=state,
            config=self._config(),
            server=_ThinkServer(_ProcessLogBuffer(), think_seconds=0.0),
            logs=logs,
            our_search_seconds=0.0,
        )

        self.assertEqual(state.opponent_think, [])
        self.assertEqual(state.opponent_think_failures, 1)
        # The battle still submitted the opponent's choice.
        self.assertEqual(bridge.messages[0]["choices"]["p1"], "move icebeam")
        self.assertNotIn("foulplay_think", state.trajectory.steps[0].metadata)


class FoulPlayThinkShardTest(unittest.TestCase):
    def _row(self, *, round_index: int, wait: float, iterations: int | None) -> dict:
        """A row in the shape `_foulplay_think_observation` actually emits."""

        row = {
            "round": round_index,
            "wait_seconds": wait,
            "iterations_per_sample": [] if iterations is None else [iterations],
            "sampled_battles": 0 if iterations is None else 1,
            "status": "miss" if iterations is None else "ok",
        }
        if iterations is None:
            row["miss_reasons"] = ["iterations_line_absent"]
            row["log_lines_scanned"] = 4
        else:
            row["budget_ms_per_sample"] = 1000
            row["iterations"] = iterations
            row["iterations_per_budget_second"] = iterations / 1.0
        return row

    def _game(self, *, battle_id: str, rows, failures: int = 0) -> ControlledFoulPlayGameResult:
        return ControlledFoulPlayGameResult(
            battle_id=battle_id,
            seed=1,
            winner="pokezero",
            pokezero_won=True,
            decision_rounds=len(rows),
            pokezero_decisions=len(rows),
            root_puct_searches=0,
            root_puct_fallbacks=0,
            opponent_think=tuple(rows),
            opponent_think_failures=failures,
        )

    def test_the_aggregate_means_over_measured_decisions_only(self) -> None:
        rows = [
            self._row(round_index=0, wait=2.0, iterations=40000),
            self._row(round_index=1, wait=1.0, iterations=20000),
            self._row(round_index=2, wait=1.5, iterations=None),
        ]

        aggregate = _foulplay_think_aggregate(rows)

        self.assertEqual(aggregate["decisions"], 3)
        self.assertEqual(aggregate["wait_measured_decisions"], 3)
        self.assertEqual(aggregate["iterations_measured_decisions"], 2)
        self.assertEqual(aggregate["total_iterations"], 60000)
        self.assertEqual(aggregate["mean_iterations"], 30000)
        self.assertEqual(aggregate["mean_iterations_per_budget_second"], 30000)
        self.assertEqual(aggregate["miss_decisions"], 1)
        self.assertEqual(aggregate["miss_reasons"], {"iterations_line_absent": 1})

    def test_coverage_is_computed_from_the_rows_not_asserted(self) -> None:
        """B4: the coverage the whole gate keys on had no failing input of its own.

        Every `iterations_coverage` in these suites used to be a hand-written literal fed to
        a header fixture, so replacing the real `len(iterations)/len(rows)` with a constant
        `1.0` broke nothing.
        """

        rows = [
            self._row(round_index=index, wait=2.0, iterations=40000 if index < 181 else None)
            for index in range(200)
        ]

        aggregate = _foulplay_think_aggregate(rows)

        self.assertEqual(aggregate["decisions"], 200)
        self.assertEqual(aggregate["iterations_measured_decisions"], 181)
        self.assertAlmostEqual(aggregate["iterations_coverage"], 0.905)

    def test_lost_decisions_are_in_the_coverage_denominator(self) -> None:
        """B6: a decision the telemetry lost appends no row, so rows-only coverage lies.

        900 of 1,000 decisions unrecorded used to read `iterations_coverage: 1.0`, which the
        cross-arm gate then compared against a healthy arm as a gap of 0.0 -- "flat".
        """

        rows = [self._row(round_index=index, wait=2.0, iterations=40000) for index in range(100)]

        honest = _foulplay_think_aggregate(rows, record_failures=900)
        blind = _foulplay_think_aggregate(rows)

        self.assertEqual(honest["decisions_attempted"], 1000)
        self.assertAlmostEqual(honest["iterations_coverage"], 0.1)
        # What it read before the denominator was fixed.
        self.assertAlmostEqual(blind["iterations_coverage"], 1.0)

    def test_the_wait_is_annotated_as_a_lower_bound_wherever_it_travels(self) -> None:
        """The annotation has to reach the merged shard, where a reader actually opens it.

        Failing input: rename the key away and nothing else notices -- which was true until
        this test existed. A mean of lower bounds is still a lower bound, and nothing beside
        `mean_wait_seconds` in the artifact says so.
        """

        aggregate = _foulplay_think_aggregate(
            [self._row(round_index=0, wait=2.0, iterations=40000)]
        )

        self.assertIn("wait_seconds_semantics", aggregate)
        self.assertIn("LOWER BOUND", aggregate["wait_seconds_semantics"])
        self.assertIn("overlapped", aggregate["wait_seconds_semantics"])

    def test_nothing_measured_reads_none_never_zero(self) -> None:
        aggregate = _foulplay_think_aggregate([])

        self.assertEqual(aggregate["decisions"], 0)
        self.assertEqual(aggregate["total_iterations"], 0)
        self.assertIsNone(aggregate["mean_iterations"])
        self.assertIsNone(aggregate["mean_iterations_per_budget_second"])

    def test_the_game_row_carries_rows_totals_and_its_own_failure_count(self) -> None:
        game = self._game(
            battle_id="battle-1",
            rows=[
                self._row(round_index=0, wait=2.0, iterations=40000),
                self._row(round_index=1, wait=1.0, iterations=20000),
            ],
            failures=2,
        )

        payload = game.to_dict()

        self.assertEqual([row["round"] for row in payload["opponent_think"]], [0, 1])
        self.assertEqual(payload["opponent_think_totals"]["total_iterations"], 60000)
        self.assertEqual(
            payload["opponent_think_totals"]["mean_iterations_per_budget_second"], 30000
        )
        self.assertEqual(payload["opponent_think_record_failures"], 2)

    def test_the_run_header_is_present_even_with_nothing_to_report(self) -> None:
        """Absent, empty and switched-off must not read the same as "no contention"."""

        result = ControlledFoulPlayBenchmarkResult(
            config=ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"), showdown_root=Path("/showdown"), games=1
            ),
            policy_id="checkpoint-raw",
            games=(),
        )

        header = result.to_dict()["foulplay_think"]

        self.assertEqual(header["schema_version"], FOULPLAY_THINK_SCHEMA_VERSION)
        self.assertEqual(header["entries_key"], "opponent_think")
        self.assertEqual(header["budget_ms_configured"], 1000)
        self.assertEqual(header["decisions"], 0)
        self.assertIsNone(header["mean_iterations_per_budget_second"])
        self.assertEqual(header["games_with_think_rows"], 0)
        self.assertEqual(header["record_failures"], 0)

    def test_the_run_header_sums_every_game_and_names_the_contention_field(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"), showdown_root=Path("/showdown"), games=2
        )
        result = ControlledFoulPlayBenchmarkResult(
            config=config,
            policy_id="checkpoint-raw",
            games=(
                self._game(
                    battle_id="battle-1",
                    rows=[self._row(round_index=0, wait=2.0, iterations=40000)],
                ),
                self._game(
                    battle_id="battle-2",
                    rows=[
                        self._row(round_index=0, wait=2.0, iterations=20000),
                        self._row(round_index=1, wait=2.0, iterations=None),
                    ],
                    failures=1,
                ),
            ),
        )

        header = result.to_dict()["foulplay_think"]

        self.assertEqual(header["decisions"], 3)
        self.assertEqual(header["iterations_measured_decisions"], 2)
        self.assertEqual(header["total_iterations"], 60000)
        # The falsifier: 40,000 visits/budget-second on one game against 20,000 on the
        # other means 30,000 for this arm, which is the number a paired eval compares.
        self.assertEqual(header["mean_iterations_per_budget_second"], 30000)
        self.assertEqual(header["miss_decisions"], 1)
        self.assertEqual(header["miss_reasons"], {"iterations_line_absent": 1})
        self.assertEqual(header["games_with_think_rows"], 2)
        self.assertEqual(header["record_failures"], 1)


class _DrainLagServer:
    """foul-play whose stdout for round 0 arrives only AFTER round 0 was accounted for.

    The drain race, in the shape that actually misattributes: the choice comes back on the
    websocket immediately, and the whole decision's stdout -- marker, `Sampling`, both
    `Iterations` lines -- lands `late_seconds` later, past the settle. Round 1 then publishes
    nothing of its own, so round 1's slice holds a complete, self-consistent record of
    round 0's work.
    """

    def __init__(self, logs: _ProcessLogBuffer, *, late_seconds: float) -> None:
        self.logs = logs
        self.late_seconds = late_seconds
        self.calls = 0
        self.sent: list[tuple[str, tuple[str, ...]]] = []

    async def send_room_lines(self, battle_id: str, lines) -> None:
        self.sent.append((battle_id, tuple(lines)))

    async def wait_for_choice(self, *, battle_id: str) -> str:
        call = self.calls
        self.calls += 1
        if call == 0:
            loop = asyncio.get_running_loop()

            def publish() -> None:
                for line in _FOULPLAY_DECISION_STDOUT:
                    self.logs.append_stdout(line)

            loop.call_later(self.late_seconds, publish)
        return "move icebeam"


def _decision_stdout(per_sample: int) -> tuple[str, ...]:
    """One decision's stdout, in foul-play's own order, with an identifiable visit count."""

    return (
        _FOULPLAY_MARKER,
        "INFO     Sampling 2 battles at 1000ms each",
        f"INFO     Iterations 0: {per_sample}",
        f"INFO     Iterations 1: {per_sample}",
    )


class _SustainedLagServer:
    """EVERY round's stdout arrives after that round was accounted for.

    Not just round 0 (`_DrainLagServer`): a CPU-heavy arm lags the drain continuously, and
    that is the case where every slice parses perfectly while belonging to the round before.
    Each round's visits are distinct so a misattributed total is identifiable by value.
    """

    def __init__(self, logs, *, late_seconds: float, per_round_visits) -> None:
        self.logs = logs
        self.late_seconds = late_seconds
        self.per_round_visits = list(per_round_visits)
        self.calls = 0
        self.sent: list[tuple[str, tuple[str, ...]]] = []

    async def send_room_lines(self, battle_id: str, lines) -> None:
        self.sent.append((battle_id, tuple(lines)))

    async def wait_for_choice(self, *, battle_id: str) -> str:
        index = self.calls
        self.calls += 1
        visits = self.per_round_visits[min(index, len(self.per_round_visits) - 1)]
        loop = asyncio.get_running_loop()

        def publish() -> None:
            for line in _decision_stdout(visits):
                self.logs.append_stdout(line)

        loop.call_later(self.late_seconds, publish)
        return "move icebeam"


class _ScriptedServer:
    """Publishes exactly what a per-round script says, optionally a little late."""

    def __init__(self, logs, *, publish, late_seconds: float | None = None) -> None:
        self.logs = logs
        self.publish = publish
        self.late_seconds = late_seconds
        self.calls = 0
        self.sent: list[tuple[str, tuple[str, ...]]] = []

    async def send_room_lines(self, battle_id: str, lines) -> None:
        self.sent.append((battle_id, tuple(lines)))

    async def wait_for_choice(self, *, battle_id: str) -> str:
        index = self.calls
        self.calls += 1
        lines = self.publish(index)

        def emit() -> None:
            for line in lines:
                self.logs.append_stdout(line)

        if self.late_seconds is None:
            emit()
        else:
            asyncio.get_running_loop().call_later(self.late_seconds, emit)
        return "move icebeam"


class _IncompleteEveryDecisionServer:
    """Every decision publishes lines, and every decision's record is unusable.

    `Sampling` declares two battles and only one `Iterations` line ever arrives, which is
    the persistent-miss shape that still produces lines -- the one the give-up guard used
    to ignore, paying the full settle on every decision forever.
    """

    def __init__(self, logs: _ProcessLogBuffer) -> None:
        self.logs = logs
        self.sent: list[tuple[str, tuple[str, ...]]] = []

    async def send_room_lines(self, battle_id: str, lines) -> None:
        self.sent.append((battle_id, tuple(lines)))

    async def wait_for_choice(self, *, battle_id: str) -> str:
        for line in (
            _FOULPLAY_MARKER,
            "INFO     Sampling 2 battles at 1000ms each",
            "INFO     Iterations 0: 41234",
        ):
            self.logs.append_stdout(line)
        return "move icebeam"


class FoulPlayThinkAttributionTest(_ThinkBoundaryDriver, unittest.TestCase):
    """B2: a decision must not report the previous decision's realized work."""

    def test_a_late_slice_is_not_attributed_to_the_next_decision(self) -> None:
        logs = _ProcessLogBuffer()
        state = self._state()

        self._run(
            state=state,
            config=self._config(),
            # Later than the 50 ms settle, so round 0 gives up and round 1 inherits.
            server=_DrainLagServer(logs, late_seconds=0.06),
            logs=logs,
            rounds=2,
            forward_request_before_round=(0, 1),
            our_search_seconds=0.08,
        )

        first, second = state.opponent_think
        self.assertEqual(first["status"], "miss")
        self.assertIn("iterations_line_absent", first["miss_reasons"])
        # The row that used to read `iterations: 81105, status: "ok"` while carrying round
        # 0's work against round 1's wait.
        self.assertEqual(second["status"], "miss")
        self.assertIn("slice_belongs_to_earlier_decision", second["miss_reasons"])
        self.assertNotIn("iterations", second)
        self.assertNotIn("iterations_per_budget_second", second)
        # The label too: a stratum read off a slice that belongs to another decision is that
        # decision's schedule, and a bucket keyed on it appears in the shard having measured
        # nothing.
        self.assertNotIn("stratum", second)
        # And the inputs, or the rate comes back as
        # `sum(iterations_per_sample) / (sampled_battles * budget_ms_per_sample / 1000)`.
        self.assertNotIn("iterations_per_sample", second)
        self.assertNotIn("sampled_battles", second)
        self.assertNotIn("budget_ms_per_sample", second)
        self.assertEqual(second["iterations_lines_seen"], 2)

    def test_a_sustained_lag_never_reports_an_earlier_decisions_work(self) -> None:
        """The case a one-round quarantine could not see, and the one lag actually produces.

        Every round's stdout arrives 60 ms after its choice -- past the settle -- so every
        slice holds exactly one decision's lines and parses perfectly. Before the cumulative
        marker count, round 2 reported round 1's visits as `ok`, round 4 reported round 3's,
        and round 4's own work was never reported at all.
        """

        logs = _ProcessLogBuffer()
        state = self._state()
        rounds = 5
        truth = [2000 * (index + 1) for index in range(rounds)]

        self._run(
            state=state,
            config=self._config(),
            server=_SustainedLagServer(logs, late_seconds=0.06, per_round_visits=truth),
            logs=logs,
            rounds=rounds,
            forward_request_before_round=tuple(range(rounds)),
            our_search_seconds=0.08,
        )

        rows = state.opponent_think
        self.assertEqual(len(rows), rounds)
        for row in rows:
            with self.subTest(round=row["round"]):
                self.assertEqual(row["status"], "miss")
                self.assertNotIn("iterations", row)
                self.assertNotIn("iterations_per_budget_second", row)
        # Specifically: no row anywhere carries any other round's realized total.
        reported = [row.get("iterations") for row in rows]
        self.assertEqual(reported, [None] * rounds)
        for row in rows[1:]:
            with self.subTest(round=row["round"]):
                self.assertIn("slice_belongs_to_earlier_decision", row["miss_reasons"])
        self.assertEqual(state.foulplay_think_decisions_seen, rounds)
        # One short, for as long as the shift lasts.
        self.assertEqual(state.foulplay_think_markers_seen, rounds - 1)
        # AND IT STOPS PAYING FOR THE SETTLE. A complete-but-misattributed record is not a
        # measurement, so keying the give-up on the PARSE would reset the streak every round
        # here and pay ~51 ms a decision forever on a run that measures nothing -- the cost
        # half of this hazard, which correctness assertions alone cannot see.
        self.assertGreaterEqual(
            state.foulplay_think_unusable_streak, _FOULPLAY_THINK_SETTLE_GIVEUP_DECISIONS
        )
        settled = [row for row in rows if "settle_seconds" in row]
        self.assertLessEqual(
            len(settled),
            _FOULPLAY_THINK_SETTLE_GIVEUP_DECISIONS,
            f"the give-up never armed: {[row.get('settle_seconds') for row in rows]}",
        )

    def test_the_instrument_self_heals_when_the_opponent_catches_up(self) -> None:
        """Refusing a shift must not disable the instrument for the rest of the battle."""

        logs = _ProcessLogBuffer()
        state = self._state()

        def publish(round_index: int) -> tuple[str, ...]:
            if round_index == 0:
                return ()                      # round 0's output is late
            if round_index == 1:
                return _decision_stdout(2000) + _decision_stdout(4000)   # catch-up: two
            return _decision_stdout(2000 * (round_index + 1))

        self._run(
            state=state,
            config=self._config(),
            server=_ScriptedServer(logs, publish=publish),
            logs=logs,
            rounds=4,
            forward_request_before_round=tuple(range(4)),
            our_search_seconds=0.0,
        )

        rows = state.opponent_think
        self.assertEqual(rows[0]["status"], "miss")
        # The catch-up slice holds two decisions and is refused as such...
        self.assertEqual(rows[1]["status"], "miss")
        self.assertIn("decision_marker_repeated", rows[1]["miss_reasons"])
        # ...but it restored the count, so measurement resumes immediately after.
        self.assertEqual(rows[2]["status"], "ok")
        self.assertEqual(rows[2]["iterations"], 2 * 2000 * 3)
        self.assertEqual(rows[3]["status"], "ok")
        self.assertEqual(state.foulplay_think_markers_seen, state.foulplay_think_decisions_seen)

    def test_the_giveup_does_not_lock_out_a_healthy_but_late_opponent(self) -> None:
        """The give-up must not manufacture the shift it exists to stop paying for.

        Three silent decisions arm it; then the opponent is healthy with a 5 ms lag -- well
        inside the settle the give-up just disabled. With no settle, every later slice holds
        the previous decision's record, and before the probation those rows read `ok` with
        stale work forever.
        """

        logs = _ProcessLogBuffer()
        state = self._state()
        rounds = 16

        def publish(round_index: int) -> tuple[str, ...]:
            if round_index < 3:
                return ()
            return _decision_stdout(2000 * (round_index + 1))

        self._run(
            state=state,
            config=self._config(),
            server=_ScriptedServer(logs, publish=publish, late_seconds=0.005),
            logs=logs,
            rounds=rounds,
            forward_request_before_round=tuple(range(rounds)),
            # PACED so each round's output lands before the NEXT round is accounted for:
            # that is what "healthy, one decision behind" means. With no pacing at all the
            # output arrives in multi-decision bursts, which is genuinely unattributable and
            # a different scenario (correctly refused, but not this one).
            our_search_seconds=0.02,
        )

        rows = state.opponent_think
        # Nothing stale is ever priced...
        for row in rows:
            with self.subTest(round=row["round"]):
                if "iterations" in row:
                    self.assertEqual(row["iterations"], 2 * 2000 * (row["round"] + 1))
        # ...and the probation gets measurement back rather than leaving the battle blind.
        recovered = [row for row in rows if row["status"] == "ok"]
        self.assertTrue(recovered, f"no decision recovered: {[r['status'] for r in rows]}")

    def test_the_giveup_arms_on_any_incomplete_parse_not_only_on_absent_lines(self) -> None:
        """B6: the four persistent-miss modes that DO produce lines used to pay forever."""

        logs = _ProcessLogBuffer()
        state = self._state()
        rounds = _FOULPLAY_THINK_SETTLE_GIVEUP_DECISIONS + 2

        self._run(
            state=state,
            config=self._config(),
            server=_IncompleteEveryDecisionServer(logs),
            logs=logs,
            rounds=rounds,
            forward_request_before_round=tuple(range(rounds)),
            our_search_seconds=0.0,
        )

        rows = state.opponent_think
        self.assertEqual(len(rows), rounds)
        for row in rows:
            with self.subTest(round=row["round"]):
                self.assertEqual(row["status"], "miss")
                self.assertIn("sample_count_mismatch", row["miss_reasons"])
                self.assertNotIn("iterations", row)
        # The give-up armed: the last rounds spent no measurable settle at all, where before
        # every one of them paid the full 50 ms.
        self.assertEqual(state.foulplay_think_unusable_streak, rounds)
        self.assertNotIn("settle_seconds", rows[-1])
        self.assertIn("settle_seconds", rows[0])


class FoulPlayThinkEarlyGameScheduleTest(_ThinkBoundaryDriver, unittest.TestCase):
    """B4, end to end: the same instrument on foul-play's OTHER schedule.

    Driven through the real parse rather than from a hand-built row, because a fixture that
    pre-supplies `stratum` cannot see the label being dropped -- and without the label the
    aggregate has nothing to stratify by, which is the whole B4 fix.
    """

    def test_the_early_game_schedule_is_labelled_and_stratified(self) -> None:
        early_stdout = (
            _FOULPLAY_MARKER,
            "INFO     Sampling 8 battles at 500ms each",
            *(f"INFO     Iterations {index}: 60000" for index in range(8)),
            "INFO     Choice: icebeam",
        )
        logs = _ProcessLogBuffer()
        state = self._state()

        self._run(
            state=state,
            config=self._config(),
            server=_ThinkServer(logs, stdout=early_stdout, think_seconds=0.0),
            logs=logs,
            our_search_seconds=0.0,
        )

        row = state.opponent_think[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["sampled_battles"], 8)
        self.assertEqual(row["budget_ms_per_sample"], 500)
        self.assertEqual(row["iterations"], 480000)
        self.assertEqual(row["stratum"], "8x500ms")
        # 480,000 visits over 8 x 0.5 s of granted budget. The SAME realized work in the
        # late-game schedule (2x1000ms) would read 240,000/s -- which is why the rate is
        # only comparable within a stratum.
        self.assertEqual(row["iterations_per_budget_second"], 120000.0)

        aggregate = _foulplay_think_aggregate(state.opponent_think)
        self.assertEqual(list(aggregate["by_stratum"]), ["8x500ms"])
        self.assertEqual(
            aggregate["by_stratum"]["8x500ms"]["mean_iterations_per_budget_second"], 120000.0
        )


class FoulPlayThinkStratumTest(unittest.TestCase):
    """B4: the rate moves on foul-play's own schedule, with zero contention."""

    def _row(self, *, round_index: int, samples: int, budget_ms: int, per_sample: int) -> dict:
        total = samples * per_sample
        return {
            "round": round_index,
            "wait_seconds": 2.0,
            "iterations_per_sample": [per_sample] * samples,
            "sampled_battles": samples,
            "budget_ms_per_sample": budget_ms,
            "iterations": total,
            "iterations_per_budget_second": total / (samples * budget_ms / 1000.0),
            "stratum": f"{samples}x{budget_ms}ms",
            "status": "ok",
        }

    def test_identical_realized_work_reads_2x_apart_across_schedules(self) -> None:
        """The demonstrated failing input for reading the unstratified mean as contention.

        480,000 realized visits either way. The early-game schedule reads half the rate of
        the late-game one, so an arm that played more early decisions looks contended and is
        not.
        """

        early = self._row(round_index=0, samples=8, budget_ms=500, per_sample=60000)
        late = self._row(round_index=1, samples=2, budget_ms=1000, per_sample=240000)

        self.assertEqual(sum(early["iterations_per_sample"]), sum(late["iterations_per_sample"]))
        self.assertEqual(early["iterations_per_budget_second"], 120000.0)
        self.assertEqual(late["iterations_per_budget_second"], 240000.0)

        aggregate = _foulplay_think_aggregate([early, late])

        # The unstratified mean is the average of two incomparable numbers...
        self.assertEqual(aggregate["mean_iterations_per_budget_second"], 180000.0)
        # ...and `by_stratum` is what makes the comparison possible at all.
        self.assertEqual(
            aggregate["by_stratum"]["8x500ms"]["mean_iterations_per_budget_second"], 120000.0
        )
        self.assertEqual(
            aggregate["by_stratum"]["2x1000ms"]["mean_iterations_per_budget_second"], 240000.0
        )

    def test_a_pure_decision_mix_difference_is_not_read_as_contention(self) -> None:
        """Two arms, identical per-stratum rates, different mixes: ratio 1.0 per stratum."""

        heavy_early = [
            self._row(round_index=i, samples=8, budget_ms=500, per_sample=60000)
            for i in range(30)
        ] + [
            self._row(round_index=30 + i, samples=2, budget_ms=1000, per_sample=240000)
            for i in range(10)
        ]
        heavy_late = [
            self._row(round_index=i, samples=8, budget_ms=500, per_sample=60000)
            for i in range(10)
        ] + [
            self._row(round_index=10 + i, samples=2, budget_ms=1000, per_sample=240000)
            for i in range(30)
        ]
        first = _foulplay_think_aggregate(heavy_early)
        second = _foulplay_think_aggregate(heavy_late)

        verdict = compare_foulplay_think(first, second, first_label="raw", second_label="search")

        self.assertEqual(verdict["status"], "ok")
        # The unstratified ratio alone would read as a 1.33x opponent speed-up on the
        # search arm out of nothing but decision mix.
        self.assertGreater(verdict["unstratified_ratio"], 1.3)
        for stratum, block in verdict["by_stratum"].items():
            with self.subTest(stratum=stratum):
                self.assertAlmostEqual(block["ratio"], 1.0)


class FoulPlayThinkGateTest(unittest.TestCase):
    """B3 and the zero-coverage gate: the comparison refuses rather than reassures."""

    def _header(
        self,
        *,
        rate: float | None,
        measured: int,
        coverage: float | None,
        strata: dict | None = None,
        observable: bool | None = True,
    ) -> dict:
        return {
            "schema_version": FOULPLAY_THINK_SCHEMA_VERSION,
            "mean_iterations_per_budget_second": rate,
            "iterations_measured_decisions": measured,
            "iterations_coverage": coverage,
            "by_stratum": strata
            if strata is not None
            else {
                "2x1000ms": {
                    "iterations_measured_decisions": measured,
                    "mean_iterations_per_budget_second": rate,
                }
            },
            "iterations_observable": observable,
            "miss_decisions": 0,
        }

    def test_null_on_both_arms_is_refused_not_read_as_flat(self) -> None:
        """The documented no-contention reading is "flat between arms". Null is not flat."""

        empty = self._header(rate=None, measured=0, coverage=0.0, strata={})

        verdict = compare_foulplay_think(empty, empty)

        self.assertEqual(verdict["status"], "refused")
        self.assertIn("a:no_rate_measured", verdict["refusal_reasons"])
        self.assertIn("a:zero_measured_decisions", verdict["refusal_reasons"])
        self.assertIn("b:no_rate_measured", verdict["refusal_reasons"])
        self.assertFalse(verdict["reading_status"]["a"]["usable"])

    def test_unequal_coverage_is_refused(self) -> None:
        """Coverage is treatment-dependent, so unequal coverage compares two subsamples.

        Under contention the CPU-heavy arm's stdout drains later, more of its decisions
        miss, and the survivors are the least-contended ones -- which pulls its rate UP,
        toward "our extra CPU did not weaken the opponent".
        """

        full = self._header(rate=450000.0, measured=100, coverage=1.0)
        thin = self._header(rate=445000.0, measured=40, coverage=0.4)

        verdict = compare_foulplay_think(full, thin)

        self.assertEqual(verdict["status"], "refused")
        self.assertIn("coverage_gap_exceeds_limit", verdict["refusal_reasons"])
        self.assertAlmostEqual(verdict["coverage_gap"], 0.6)
        self.assertGreater(verdict["coverage_gap"], FOULPLAY_THINK_MAX_COVERAGE_GAP)

    def test_no_shared_stratum_is_refused(self) -> None:
        early_only = self._header(
            rate=120000.0,
            measured=50,
            coverage=1.0,
            strata={
                "8x500ms": {
                    "decisions": 50,
                    "iterations_measured_decisions": 50,
                    "mean_iterations_per_budget_second": 120000.0,
                }
            },
        )
        late_only = self._header(rate=240000.0, measured=50, coverage=1.0)

        verdict = compare_foulplay_think(early_only, late_only)

        self.assertEqual(verdict["status"], "refused")
        self.assertIn("no_shared_stratum_with_measured_rate", verdict["refusal_reasons"])
        # And the unstratified ratio it would otherwise have reported is a factor of two.
        self.assertAlmostEqual(verdict["unstratified_ratio"], 2.0)

    def test_a_handful_of_measured_decisions_is_refused(self) -> None:
        thin = self._header(rate=450000.0, measured=3, coverage=0.06)

        verdict = compare_foulplay_think(thin, thin)

        self.assertEqual(verdict["status"], "refused")
        self.assertIn("a:below_minimum_measured_decisions", verdict["refusal_reasons"])
        self.assertLess(3, FOULPLAY_THINK_MIN_MEASURED_DECISIONS)

    def test_an_unobservable_start_method_is_refused(self) -> None:
        """`spawn`/`forkserver` cannot emit the line at all, so there is nothing to read."""

        blind = self._header(rate=None, measured=0, coverage=0.0, strata={}, observable=False)

        status = foulplay_think_reading_status(blind)

        self.assertFalse(status["usable"])
        self.assertIn("opponent_start_method_cannot_emit_iterations", status["reasons"])

    def test_lost_decisions_make_the_reading_unusable(self) -> None:
        """The same B6 hazard, at the gate: a shrunken denominator must not pass as clean."""

        blind = self._header(rate=450000.0, measured=100, coverage=1.0)
        blind["record_failures"] = 900

        status = foulplay_think_reading_status(blind)

        self.assertFalse(status["usable"])
        self.assertIn("telemetry_record_failures", status["reasons"])

    def test_an_absent_block_is_refused_by_name(self) -> None:
        status = foulplay_think_reading_status(None)

        self.assertFalse(status["usable"])
        self.assertEqual(status["reasons"], ["think_block_absent"])

    def test_a_stratum_too_thin_to_compare_is_excluded_and_named(self) -> None:
        """A ratio backed by one decision must not read like one backed by hundreds."""

        thick = self._header(rate=450000.0, measured=100, coverage=1.0)
        thick["by_stratum"]["8x500ms"] = {
            "iterations_measured_decisions": 60,
            "mean_iterations_per_budget_second": 120000.0,
        }
        thin = self._header(rate=445000.0, measured=98, coverage=0.98)
        thin["by_stratum"]["8x500ms"] = {
            # ONE decision, and by chance it looks halved -- exactly the shape that would
            # otherwise be quoted beside a 60-decision stratum as if equally solid.
            "iterations_measured_decisions": 1,
            "mean_iterations_per_budget_second": 60000.0,
        }

        verdict = compare_foulplay_think(thick, thin, first_label="raw", second_label="search")

        self.assertNotIn("8x500ms", verdict["by_stratum"])
        self.assertEqual(verdict["thin_strata"], ["8x500ms"])
        self.assertLess(1, FOULPLAY_THINK_MIN_STRATUM_DECISIONS)
        # The stratum that IS thick enough still reports, with its n beside the ratio.
        self.assertEqual(
            verdict["by_stratum"]["2x1000ms"]["raw_iterations_measured_decisions"], 100
        )
        self.assertEqual(
            verdict["by_stratum"]["2x1000ms"]["search_iterations_measured_decisions"], 98
        )

    def test_every_reported_ratio_carries_its_denominator(self) -> None:
        raw = self._header(rate=452500.0, measured=100, coverage=1.0)
        search = self._header(rate=119000.0, measured=98, coverage=0.98)

        verdict = compare_foulplay_think(raw, search, first_label="raw", second_label="search")

        for stratum, block in verdict["by_stratum"].items():
            with self.subTest(stratum=stratum):
                self.assertIn("raw_iterations_measured_decisions", block)
                self.assertIn("search_iterations_measured_decisions", block)

    def _real_header(self, *, rows, record_failures: int = 0) -> dict:
        """A header built by the REAL aggregate over rows in the shape the row emitter emits.

        The gate tests otherwise run on hand-written headers, which cannot see the aggregate
        and the gate disagreeing about a field's meaning.
        """

        aggregate = _foulplay_think_aggregate(rows, record_failures=record_failures)
        return {
            "schema_version": FOULPLAY_THINK_SCHEMA_VERSION,
            **aggregate,
            "iterations_observable": True,
            "record_failures": record_failures,
        }

    @staticmethod
    def _stratum_row(round_index: int, *, samples: int, budget_ms: int, per_sample: int) -> dict:
        total = samples * per_sample
        return {
            "round": round_index,
            "wait_seconds": 2.0,
            "iterations_per_sample": [per_sample] * samples,
            "sampled_battles": samples,
            "budget_ms_per_sample": budget_ms,
            "iterations": total,
            "iterations_per_budget_second": total / (samples * budget_ms / 1000.0),
            "stratum": f"{samples}x{budget_ms}ms",
            "status": "ok",
        }

    def test_a_sliver_of_shared_stratum_cannot_certify_the_whole_run(self) -> None:
        """B5: 500-vs-1 in each stratum used to return `ok` with both ratios 1.0.

        Each arm sits almost entirely in a stratum the other visited once, so the two ratios
        are computed off n=1 while 99.8% of each arm is never compared like-for-like.
        """

        raw_rows = [
            self._stratum_row(i, samples=2, budget_ms=1000, per_sample=225000) for i in range(500)
        ] + [self._stratum_row(500, samples=8, budget_ms=500, per_sample=60000)]
        search_rows = [
            self._stratum_row(i, samples=8, budget_ms=500, per_sample=60000) for i in range(500)
        ] + [self._stratum_row(500, samples=2, budget_ms=1000, per_sample=225000)]

        verdict = compare_foulplay_think(
            self._real_header(rows=raw_rows),
            self._real_header(rows=search_rows),
            first_label="raw",
            second_label="search",
        )

        self.assertEqual(verdict["status"], "refused")
        # The thin strata are excluded by name, which leaves nothing to compare at all.
        self.assertEqual(verdict["thin_strata"], ["2x1000ms", "8x500ms"])
        self.assertIn("no_shared_stratum_with_measured_rate", verdict["refusal_reasons"])

    def test_a_real_starvation_hidden_behind_a_sliver_is_refused(self) -> None:
        """The same shape with an actual 3.8x drop, which used to read `ratio: 1.0`, ok."""

        raw_rows = [
            self._stratum_row(i, samples=2, budget_ms=1000, per_sample=225000) for i in range(200)
        ]
        search_rows = [
            self._stratum_row(i, samples=8, budget_ms=500, per_sample=29500) for i in range(199)
        ] + [self._stratum_row(199, samples=2, budget_ms=1000, per_sample=225000)]

        verdict = compare_foulplay_think(
            self._real_header(rows=raw_rows),
            self._real_header(rows=search_rows),
            first_label="raw",
            second_label="search",
        )

        self.assertEqual(verdict["status"], "refused")
        self.assertTrue(
            any("compared_strata_cover_too_little" in reason for reason in verdict["refusal_reasons"])
            or "no_shared_stratum_with_measured_rate" in verdict["refusal_reasons"],
            verdict["refusal_reasons"],
        )
        # The unstratified ratio it would have reported does show the drop -- but it is the
        # number that also moves 2x on schedule alone, which is why it cannot carry a verdict.
        self.assertLess(verdict["unstratified_ratio"], 0.3)

    def test_a_thick_shared_stratum_holding_a_sliver_of_each_arm_is_refused(self) -> None:
        """The share floor, on the case the thin-stratum floor does NOT catch.

        The compared stratum is well-powered on both sides (10 vs 500), so it is not thin --
        but it holds 2% of the raw arm, whose other 500 decisions sit in a stratum the search
        arm visited twice. A clean ratio off 2% of an arm is not a statement about the arm.
        """

        raw_rows = [
            self._stratum_row(i, samples=2, budget_ms=1000, per_sample=225000) for i in range(500)
        ] + [
            self._stratum_row(500 + i, samples=8, budget_ms=500, per_sample=60000)
            for i in range(10)
        ]
        search_rows = [
            self._stratum_row(i, samples=8, budget_ms=500, per_sample=60000) for i in range(500)
        ] + [
            self._stratum_row(500 + i, samples=2, budget_ms=1000, per_sample=225000)
            for i in range(2)
        ]

        verdict = compare_foulplay_think(
            self._real_header(rows=raw_rows),
            self._real_header(rows=search_rows),
            first_label="raw",
            second_label="search",
        )

        self.assertEqual(verdict["status"], "refused")
        # The 500-vs-2 stratum is excluded as thin; the 10-vs-500 one IS compared...
        self.assertEqual(verdict["thin_strata"], ["2x1000ms"])
        self.assertIn("8x500ms", verdict["by_stratum"])
        # ...and covers 2% of the raw arm, which is what the share floor refuses.
        self.assertIn("raw:compared_strata_cover_too_little", verdict["refusal_reasons"])
        self.assertLess(verdict["compared_share"]["raw"], 0.05)
        self.assertGreater(verdict["compared_share"]["search"], 0.9)

    def test_the_bulk_share_is_reported_on_every_verdict(self) -> None:
        rows = [
            self._stratum_row(i, samples=2, budget_ms=1000, per_sample=225000) for i in range(100)
        ]
        verdict = compare_foulplay_think(
            self._real_header(rows=rows), self._real_header(rows=rows),
            first_label="raw", second_label="search",
        )

        self.assertEqual(verdict["status"], "ok")
        self.assertAlmostEqual(verdict["compared_share"]["raw"], 1.0)
        self.assertAlmostEqual(verdict["compared_share"]["search"], 1.0)

    def test_a_comparable_pair_is_accepted_and_reports_the_drop(self) -> None:
        """The instrument working: same strata, same coverage, opponent visibly slowed."""

        raw = self._header(rate=452500.0, measured=100, coverage=1.0)
        search = self._header(rate=119000.0, measured=98, coverage=0.98)
        search["by_stratum"]["2x1000ms"]["mean_iterations_per_budget_second"] = 119000.0

        verdict = compare_foulplay_think(raw, search, first_label="raw", second_label="search")

        self.assertEqual(verdict["status"], "ok")
        self.assertEqual(verdict["refusal_reasons"], [])
        self.assertAlmostEqual(verdict["by_stratum"]["2x1000ms"]["ratio"], 119000.0 / 452500.0)
        self.assertLess(verdict["by_stratum"]["2x1000ms"]["ratio"], 0.3)


class FoulPlayStartMethodProbeTest(unittest.TestCase):
    """The one fact the measurement rests on, recorded instead of assumed."""

    def test_the_probe_reads_this_interpreters_real_start_method(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            foulplay_python=Path(sys.executable),
        )

        probe = asyncio.run(_probe_foulplay_start_method(config))

        self.assertEqual(probe["status"], "probed")
        self.assertIn(probe["opponent_start_method"], {"fork", "spawn", "forkserver"})
        # And it resolves to the boolean the gate reads. On this machine (macOS) the answer
        # is False: `spawn` cannot emit the line at all, which is the demonstrated failing
        # input for `iterations_observable` -- no mock required.
        self.assertEqual(
            probe["iterations_observable"], probe["opponent_start_method"] == "fork"
        )

    def test_a_missing_interpreter_records_unknown_not_observable(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            foulplay_python=Path("/nonexistent/python-that-is-not-there"),
        )

        probe = asyncio.run(_probe_foulplay_start_method(config))

        self.assertEqual(probe["status"], "probe_failed")
        self.assertIsNone(probe["iterations_observable"])

    def test_the_probe_ignores_noise_before_its_answer(self) -> None:
        """A wrapper or a warning printing first used to be read AS the start method."""

        with tempfile.TemporaryDirectory() as tmp:
            noisy = Path(tmp) / "noisy-python"
            noisy.write_text(
                "#!/bin/sh\n"
                "echo 'warning: venv is stale'\n"
                f"exec {sys.executable} \"$@\"\n"
            )
            noisy.chmod(0o755)
            config = ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                foulplay_python=noisy,
            )

            probe = asyncio.run(_probe_foulplay_start_method(config))

        self.assertEqual(probe["status"], "probed")
        self.assertIn(probe["opponent_start_method"], {"fork", "spawn", "forkserver"})
        self.assertNotEqual(probe["opponent_start_method"], "warning:")

    def test_a_hanging_interpreter_times_out_fast_and_leaks_nothing(self) -> None:
        """Two failing inputs in one: a leaked child, and a cleanup that stalls the run.

        The first cleanup I wrote killed the child and then awaited it UNBOUNDED, which
        blocked for the child's natural lifetime -- measured 30.1 s against a 0.3 s timeout.
        A stalled run is worse than the orphan it was fixing, so both waits are bounded and
        the whole process group is taken out (a wrapper script's child otherwise survives its
        parent and keeps holding our pipe).
        """

        marker = "sleep 31.5"
        with tempfile.TemporaryDirectory() as tmp:
            hanging = Path(tmp) / "hanging-python"
            hanging.write_text(f"#!/bin/sh\n{marker}\n")
            hanging.chmod(0o755)
            config = ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                foulplay_python=hanging,
            )

            started = time.monotonic()
            probe = asyncio.run(_probe_foulplay_start_method(config, timeout_seconds=0.3))
            elapsed = time.monotonic() - started

        self.assertEqual(probe["status"], "probe_failed")
        self.assertIsNone(probe["iterations_observable"])
        self.assertIn("TimeoutError", probe["error"])
        # Returns at its timeout, not at the child's convenience.
        self.assertLess(elapsed, 3.0)
        self.assertTrue(probe["probe_child_reaped"])
        # And the wrapper's own child does not outlive it.
        survivors = subprocess.run(
            ["pgrep", "-f", marker], capture_output=True, text=True
        ).stdout.split()
        self.assertEqual(survivors, [])

    def test_a_child_that_escapes_the_process_group_cannot_stall_the_reap(self) -> None:
        """The failing input for BOUNDING the reap, as opposed to just killing the group.

        A wrapper that daemonises a helper leaves a grandchild in its own session holding the
        inherited stdout pipe, which `killpg` on our group cannot reach. Measured on that
        input: an unbounded `await process.wait()` blocks 31.22 s, the bounded one returns in
        1.00 s and says plainly that it could not reap.
        """

        marker = "sleep-escapee-31.5"
        script = (
            "#!/bin/sh\n"
            f"{sys.executable} -c 'import os,time; os.setsid(); time.sleep(31.5)' "
            f"# {marker}\n"
            "sleep 31.5\n"
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wrapper = Path(tmp) / "wrapper-python"
                wrapper.write_text(script)
                wrapper.chmod(0o755)
                config = ControlledFoulPlayConfig(
                    checkpoint=Path("checkpoint.pt"),
                    showdown_root=Path("/showdown"),
                    foulplay_python=wrapper,
                )

                started = time.monotonic()
                probe = asyncio.run(_probe_foulplay_start_method(config, timeout_seconds=0.3))
                elapsed = time.monotonic() - started

            self.assertEqual(probe["status"], "probe_failed")
            self.assertLess(elapsed, 5.0)
            self.assertIn("probe_child_reaped", probe)
        finally:
            subprocess.run(["pkill", "-f", "31.5"], capture_output=True)

    def test_the_run_actually_probes(self) -> None:
        """Deleting the probe call left all tests green, so nothing pinned that it runs."""

        import inspect

        import pokezero.foulplay_bridge as bridge

        # `_run_controlled_foulplay_games` is the function that owns the game loop; both
        # public entry points funnel into it, so pinning it here covers the audit driver too.
        source = inspect.getsource(bridge._run_controlled_foulplay_games)
        self.assertIn("foulplay_start_method = await _probe_foulplay_start_method(config)", source)
        # Only when there IS a foul-play process to probe.
        self.assertLess(
            source.index("if opponent_policy is None:"),
            source.index("_probe_foulplay_start_method(config)"),
        )
        # And it reaches the result the summary is built from.
        self.assertIn("foulplay_start_method=foulplay_start_method", source)

    def test_the_header_carries_the_probe_and_its_verdict(self) -> None:
        result = ControlledFoulPlayBenchmarkResult(
            config=ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"), showdown_root=Path("/showdown"), games=1
            ),
            policy_id="checkpoint-raw",
            games=(),
            foulplay_start_method={
                "status": "probed",
                "opponent_start_method": "forkserver",
                "opponent_python": "3.14.0",
                "iterations_observable": False,
            },
        )

        header = result.to_dict()["foulplay_think"]

        self.assertEqual(header["opponent_start_method"], "forkserver")
        self.assertIs(header["iterations_observable"], False)
        self.assertFalse(header["reading"]["usable"])
        self.assertIn(
            "opponent_start_method_cannot_emit_iterations", header["reading"]["reasons"]
        )


class FoulPlayThinkClockSourceTest(unittest.TestCase):
    """The `monotonic` claim, pinned where it can actually fail.

    A passing test suite was offered as evidence for this and could not have caught the
    violation: swapping every telemetry stamp to `perf_counter` leaves the legacy boundary
    test green, because its two pinned readings run out and the third raises inside the
    instrument's own counted try/except. So the invariant is asserted on the source.
    """

    @staticmethod
    def _clock_calls(function) -> set[str]:
        """Clock attribute names actually CALLED, read off the AST.

        Not a substring check: both of these functions explain in prose why they avoid
        `perf_counter`, and a docstring that names the trap must not be mistaken for the
        trap. The AST sees code only.
        """

        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        return {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"monotonic", "perf_counter", "time"}
        }

    def test_no_telemetry_stamp_consumes_a_perf_counter_reading(self) -> None:
        import pokezero.foulplay_bridge as bridge

        for function in (
            bridge._foulplay_think_observation,
            bridge._wait_for_foulplay_choice_or_exit,
        ):
            with self.subTest(function=function.__name__):
                calls = self._clock_calls(function)
                self.assertIn("monotonic", calls)
                self.assertNotIn("perf_counter", calls)

    def test_the_request_forward_stamp_is_monotonic_too(self) -> None:
        import inspect

        import pokezero.foulplay_bridge as bridge

        source = inspect.getsource(bridge._handle_stream_event)
        self.assertIn("state.foulplay_request_forwarded_monotonic = time.monotonic()", source)
        self.assertNotIn("perf_counter", self._clock_calls(bridge._handle_stream_event))


if __name__ == "__main__":
    unittest.main()
