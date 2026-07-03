from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.foulplay_bridge import (
    ControlledFoulPlayBenchmarkResult,
    ControlledFoulPlayComparisonResult,
    ControlledFoulPlayConfig,
    ControlledFoulPlayGameResult,
    _ControlledBattleState,
    _choice_body_from_outgoing_message,
    _build_policy,
    _foulplay_command,
    _foulplay_env,
    _line_for_foulplay,
    _line_chunks_safe_for_foulplay,
    _observation_with_search_metadata,
    _root_puct_prior_action_change_details,
    _requested_legal_action_masks_for_context,
    _is_terminal_protocol_line,
    _split_outgoing_showdown_message,
    _terminal_line_for_foulplay,
    _write_json,
    async_comparison_main,
    async_main,
    build_arg_parser,
    build_comparison_arg_parser,
    run_controlled_foulplay_comparison,
    run_controlled_foulplay_benchmark,
)
from pokezero.env import TerminalState
from pokezero.observation import PokeZeroObservationV0
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
        self.assertIsNone(config.root_prior_temperature)
        self.assertEqual(config.effective_root_prior_temperature, 1.0)
        self.assertEqual(config.root_opponent_action_scenarios, 1)
        self.assertEqual(config.root_opponent_action_candidate_scenarios, ACTION_COUNT)
        self.assertEqual(args.selection_mode, "visits")
        self.assertEqual(args.root_visit_budget, 16)
        self.assertIsNone(args.root_prior_temperature)
        self.assertEqual(args.root_opponent_action_scenarios, 1)
        self.assertEqual(args.root_opponent_action_candidate_scenarios, ACTION_COUNT)

        warmed_config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            temperature=1.75,
        )
        self.assertEqual(warmed_config.effective_root_prior_temperature, 1.75)

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
                env_config=object(),
                rollout_config=object(),
                policy_id="fake-base",
            )

        self.assertEqual(
            getattr(policy.opponent_action_scenario_planner, "planner_id"),
            f"checkpoint-top{ACTION_COUNT}",
        )
        self.assertEqual(policy.max_opponent_action_scenarios, 1)

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
                    root_puct_selected_prior_action_changes=2,
                    root_puct_pre_gate_prior_action_changes=3,
                    root_puct_time_budget_exhaustions=2,
                    root_puct_start_override_sources_used=3,
                    root_puct_start_override_attempts_used=5,
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
                    root_puct_selected_prior_action_changes=1,
                    root_puct_pre_gate_prior_action_changes=2,
                    root_puct_time_budget_exhaustions=1,
                    root_puct_start_override_sources_used=1,
                    root_puct_start_override_attempts_used=4,
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
        self.assertEqual(payload["root_puct"]["opponent_action_scenarios_unsearched"], 3)
        self.assertEqual(payload["root_puct"]["selected_prior_action_changes"], 3)
        self.assertEqual(payload["root_puct"]["pre_gate_prior_action_changes"], 5)
        self.assertEqual(payload["root_puct"]["time_budget_exhaustions"], 3)
        self.assertEqual(payload["root_puct"]["start_override_sources_used"], 4)
        self.assertEqual(payload["root_puct"]["start_override_attempts"], 7)
        self.assertEqual(payload["root_puct"]["start_override_attempts_used"], 9)
        self.assertEqual(payload["root_puct"]["fallback_reasons"], {"search failed: boom": 2})
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_scenarios_generated"], 9)
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_scenarios_skipped"], 1)
        self.assertEqual(payload["game_results"][0]["root_puct_opponent_action_scenarios_unsearched"], 2)
        self.assertEqual(payload["game_results"][0]["root_puct_selected_prior_action_changes"], 2)
        self.assertEqual(payload["game_results"][0]["root_puct_pre_gate_prior_action_changes"], 3)
        self.assertEqual(payload["game_results"][0]["root_puct_time_budget_exhaustions"], 2)
        self.assertEqual(payload["game_results"][0]["root_puct_start_override_sources_used"], 3)
        self.assertEqual(payload["game_results"][0]["root_puct_start_override_attempts_used"], 5)
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
        self.assertEqual(payload["root_puct"]["minimum_score_improvement"], 0.1)
        self.assertEqual(payload["root_puct"]["root_prior_temperature"], 2.5)
        self.assertEqual(payload["root_puct"]["root_visit_budget"], 16)
        self.assertEqual(payload["root_puct"]["root_time_budget_ms"], 250)
        self.assertEqual(payload["root_puct"]["root_opponent_action_scenarios"], 2)
        self.assertEqual(payload["root_puct"]["root_opponent_action_candidate_scenarios"], 5)
        self.assertEqual(payload["root_puct"]["leaf_rollout_sampling"], True)
        self.assertEqual(payload["root_puct"]["belief_start_overrides"], True)
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
                "--summary-out",
                str(summary_path),
            )
            with patch(
                "pokezero.foulplay_bridge.run_controlled_foulplay_comparison",
                side_effect=fake_comparison,
            ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = asyncio.run(async_comparison_main(argv))

            payload = json.loads(summary_path.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema_version"], "pokezero.controlled-foulplay-comparison.v1")
        self.assertEqual(payload["comparison_mode"], "per-seed")
        self.assertEqual(payload["comparison"]["paired_by_seed"]["root_puct"]["wins"], 1)
        self.assertEqual(build_comparison_arg_parser().parse_args(argv).games, 1)
        self.assertEqual(build_comparison_arg_parser().parse_args(argv).comparison_mode, "per-seed")
        self.assertIn("DIAGNOSTIC RESULT", stdout.getvalue())
        self.assertIn("(per-seed)", stdout.getvalue())
        self.assertIn("descriptive_delta=100.0%", stdout.getvalue())

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
