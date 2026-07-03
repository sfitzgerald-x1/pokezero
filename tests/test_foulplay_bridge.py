from __future__ import annotations

import asyncio
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pokezero.foulplay_bridge import (
    ControlledFoulPlayBenchmarkResult,
    ControlledFoulPlayConfig,
    ControlledFoulPlayGameResult,
    _ControlledBattleState,
    _choice_body_from_outgoing_message,
    _foulplay_command,
    _foulplay_env,
    _line_for_foulplay,
    _line_chunks_safe_for_foulplay,
    _root_puct_prior_action_change_details,
    _requested_legal_action_masks_for_context,
    _is_terminal_protocol_line,
    _split_outgoing_showdown_message,
    _terminal_line_for_foulplay,
    _write_json,
    async_main,
    run_controlled_foulplay_benchmark,
)
from pokezero.env import TerminalState
from pokezero.policy import PolicyDecision


class FoulPlayBridgeTest(unittest.TestCase):
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

    def test_is_terminal_protocol_line_detects_win_and_tie(self) -> None:
        self.assertTrue(_is_terminal_protocol_line("|win|PokeZeroBot"))
        self.assertTrue(_is_terminal_protocol_line("|tie|"))
        self.assertFalse(_is_terminal_protocol_line("|turn|2"))

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
        with self.assertRaisesRegex(ValueError, "root_visit_budget"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                root_visit_budget=0,
            )
        with self.assertRaisesRegex(ValueError, "leaf_rollout_sampling"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                leaf_rollout_sampling=True,
            )
        with self.assertRaisesRegex(ValueError, "foulplay_random_seed"):
            ControlledFoulPlayConfig(
                checkpoint=Path("checkpoint.pt"),
                showdown_root=Path("/showdown"),
                foulplay_random_seed=-1,
            )

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
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=2,
            selection_mode="visits",
            minimum_value_improvement=0.25,
            minimum_override_prior_ratio=0.5,
            root_visit_budget=16,
            leaf_rollout_rounds=1,
            leaf_rollout_sampling=True,
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
                    root_puct_selected_prior_action_changes=2,
                    root_puct_pre_gate_prior_action_changes=3,
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
                    root_puct_selected_prior_action_changes=1,
                    root_puct_pre_gate_prior_action_changes=2,
                    root_puct_fallback_reasons={"search failed: boom": 2},
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
        self.assertEqual(payload["root_puct"]["selected_prior_action_changes"], 3)
        self.assertEqual(payload["root_puct"]["pre_gate_prior_action_changes"], 5)
        self.assertEqual(payload["root_puct"]["fallback_reasons"], {"search failed: boom": 2})
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_scenarios_generated"], 9)
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_scenarios_skipped"], 1)
        self.assertEqual(payload["game_results"][0]["root_puct_selected_prior_action_changes"], 2)
        self.assertEqual(payload["game_results"][0]["root_puct_pre_gate_prior_action_changes"], 3)
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
            {"search failed: boom": 2},
        )
        self.assertEqual(payload["root_puct"]["opponent_legal_mask_mode"], "hidden")
        self.assertEqual(payload["root_puct"]["foulplay_search_time_ms"], 1000)
        self.assertEqual(payload["root_puct"]["selection_mode"], "visits")
        self.assertEqual(payload["root_puct"]["minimum_value_improvement"], 0.25)
        self.assertEqual(payload["root_puct"]["minimum_override_prior_ratio"], 0.5)
        self.assertEqual(payload["root_puct"]["root_visit_budget"], 16)
        self.assertEqual(payload["root_puct"]["leaf_rollout_sampling"], True)
        self.assertAlmostEqual(payload["root_puct"]["average_elapsed_seconds"], 0.3)

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

    def test_run_controlled_foulplay_benchmark_emits_incremental_progress(self) -> None:
        class FakeModelConfig:
            policy_id = "checkpoint"
            categorical_feature_count = 1
            numeric_feature_count = 1

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

        async def run_single_game(**_: object) -> ControlledFoulPlayGameResult:
            return next(game_results)

        async def spawn_foulplay(*_: object, **__: object) -> FakeProcess:
            return FakeProcess()

        with (
            patch("pokezero.foulplay_bridge._validate_external_paths"),
            patch("pokezero.foulplay_bridge.load_transformer_checkpoint", return_value=(object(), FakeCheckpointResult())),
            patch("pokezero.foulplay_bridge.gen3_category_vocabulary", return_value=object()),
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

    def test_async_main_summary_out_preserves_partial_progress_on_failure(self) -> None:
        class FakeModelConfig:
            policy_id = "checkpoint"
            categorical_feature_count = 1
            numeric_feature_count = 1

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
                patch("pokezero.foulplay_bridge.gen3_category_vocabulary", return_value=object()),
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


if __name__ == "__main__":
    unittest.main()
