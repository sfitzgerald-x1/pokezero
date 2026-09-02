from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult, TerminalState
from pokezero import foulplay_bridge
from pokezero.foulplay_bridge import ControlledFoulPlayConfig, _BattleBridge, _ControlledBattleState
from pokezero.live_foulplay_continuation import (
    LiveFoulPlayBoundary,
    LiveFoulPlayContinuationBoundExceeded,
    LiveFoulPlayContinuationError,
    reconstruct_live_foulplay_boundary,
    run_live_foulplay_continuation,
    select_live_foulplay_continuation_oracle_action,
)
from pokezero.rollout import RolloutConfig
from pokezero.trajectory import BattleTrajectory


class _FakeEnv:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.observations = {"p1": object(), "p2": object()}

    def reset(self, *, seed: int, format_id: str) -> None:
        self.calls.append(("reset", seed, format_id))

    def restore(self, snapshot: object) -> None:
        self.calls.append(("restore", snapshot))

    def requested_players(self) -> tuple[str, str]:
        return ("p1", "p2")

    def step(self, actions: dict[str, int]) -> StepResult:
        self.calls.append(("step", dict(actions)))
        return StepResult(observations=self.observations, rewards={}, terminal=None)

    def close(self) -> None:
        self.calls.append("close")


class _FakeTerminalEnv(_FakeEnv):
    def step(self, actions: dict[str, int]) -> StepResult:
        self.calls.append(("step", dict(actions)))
        return StepResult(
            observations={},
            rewards={},
            terminal=TerminalState(winner="p2", turn_count=4, capped=False),
        )


class _FakeSourceBridge:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def send(self, command: dict[str, object]) -> None:
        if command.get("type") == "choices":
            self.calls.append("source-choices")


class _FakeProbe:
    def __init__(self, calls: list[str], *, failure: Exception | None = None) -> None:
        self.calls = calls
        self.failure = failure
        self.boundary = object()

    async def capture_boundary_async(self, **_kwargs: object) -> object:
        self.calls.append("snapshot-bound")
        return self.boundary

    def run(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["boundary"] is self.boundary
        assert kwargs["decision_round"] == 1
        assert kwargs["pokezero_action"] == 0
        assert kwargs["foulplay_action"] == 1
        assert kwargs["foulplay_choice"] == "move 2"
        self.calls.append("local-continuation")
        if self.failure is not None:
            raise self.failure
        return {
            "continuation": {"decision_round_count": 1},
            "source_request_sha256": {"p1": "a", "p2": "b"},
            "snapshot_request_sha256": {"p1": "c", "p2": "d"},
        }


class _FakeOracleController:
    def __init__(self, calls: list[str], *, failure: Exception | None = None) -> None:
        self.calls = calls
        self.failure = failure
        self.boundary = object()

    async def capture_boundary_async(self, **_kwargs: object) -> object:
        self.calls.append("oracle-snapshot-bound")
        return self.boundary

    def select(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["boundary"] is self.boundary
        assert kwargs["raw_action"] == 0
        assert kwargs["legal_actions"] == (0, 1)
        assert kwargs["foulplay_action"] == 1
        self.calls.append("oracle-controller")
        if self.failure is not None:
            raise self.failure
        return {
            "action_index": 1,
            "controller": "live-foulplay-continuation-oracle",
            "controller_status": "oracle-selected",
        }


class LiveFoulPlayContinuationTest(unittest.TestCase):
    def test_oracle_config_is_raw_external_foulplay_only_and_keeps_both_orientations(self) -> None:
        for seat in ("p1", "p2"):
            with self.subTest(seat=seat):
                config = ControlledFoulPlayConfig(
                    checkpoint=Path("/tmp/ckpt.pt"),
                    showdown_root=Path("/tmp/showdown"),
                    policy_mode="raw",
                    opponent_policy_mode="foul-play",
                    pokezero_player=seat,
                    live_continuation_oracle=True,
                    live_continuation_oracle_candidate_cap=9,
                    max_decision_rounds=8,
                )
                self.assertTrue(config.live_continuation_oracle)
                self.assertEqual(config.foulplay_player, "p2" if seat == "p1" else "p1")
        with self.assertRaisesRegex(ValueError, "raw PokeZero source policy"):
            ControlledFoulPlayConfig(
                checkpoint=Path("/tmp/ckpt.pt"),
                showdown_root=Path("/tmp/showdown"),
                policy_mode="root-puct",
                live_continuation_oracle=True,
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            ControlledFoulPlayConfig(
                checkpoint=Path("/tmp/ckpt.pt"),
                showdown_root=Path("/tmp/showdown"),
                policy_mode="raw",
                live_continuation_oracle=True,
                live_continuation_oracle_max_continuation_decision_rounds=0,
            )
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            ControlledFoulPlayConfig(
                checkpoint=Path("/tmp/ckpt.pt"),
                showdown_root=Path("/tmp/showdown"),
                policy_mode="raw",
                live_continuation_oracle=True,
                live_continuation_oracle_progress_dir=Path("relative-progress"),
            )

    def test_oracle_progress_recorder_is_create_only_and_snapshot_free(self) -> None:
        with TemporaryDirectory() as directory:
            recorder = foulplay_bridge._LiveFoulPlayContinuationOracleProgressRecorder(
                root=Path(directory)
            )
            recorder.record(
                {
                    "event": "candidate-started",
                    "source_decision_round": 2,
                    "candidate_index": 0,
                    "candidate_count": 2,
                    "action_index": 1,
                    "max_continuation_decision_rounds": 128,
                }
            )
            path = Path(directory) / "00000000-candidate-started.json"
            original_bytes = path.read_bytes()
            record = json.loads(original_bytes)
            self.assertEqual(record["schema_version"], "pokezero.live-foulplay-continuation-oracle-progress.v1")
            self.assertEqual(record["action_index"], 1)
            self.assertNotIn("snapshot", record)
            with self.assertRaisesRegex(LiveFoulPlayContinuationError, "unexpected schema"):
                recorder.record({"event": "candidate-started", "snapshot": {"secret": True}})
            collision = foulplay_bridge._LiveFoulPlayContinuationOracleProgressRecorder(
                root=Path(directory)
            )
            with self.assertRaisesRegex(LiveFoulPlayContinuationError, "already exists"):
                collision.record(
                    {
                        "event": "candidate-started",
                        "source_decision_round": 2,
                        "candidate_index": 0,
                        "candidate_count": 2,
                        "action_index": 0,
                        "max_continuation_decision_rounds": 128,
                    }
                )
            self.assertEqual(path.read_bytes(), original_bytes)
            with self.assertRaisesRegex(LiveFoulPlayContinuationError, "terminal_winner is invalid"):
                recorder.record(
                    {
                        "event": "candidate-completed",
                        "source_decision_round": 2,
                        "candidate_index": 0,
                        "candidate_count": 2,
                        "action_index": 1,
                        "continuation_decision_round_count": 1,
                        "terminal_after_fixed_joint_step": False,
                        "terminal_winner": {"snapshot": "forbidden"},
                        "elapsed_milliseconds": 1,
                        "max_continuation_decision_rounds": 128,
                    }
                )

    def _run_oracle_boundary_seam(
        self, *, failure: Exception | None = None, decision_round: int = 1
    ) -> tuple[list[str], _ControlledBattleState]:
        calls: list[str] = []
        state = _ControlledBattleState(
            battle_id="live-oracle-108000000",
            seed=108_000_000,
            format_id="gen3randombattle",
            request_lines={"p1": "|request|{}", "p2": "|request|{}"},
            trajectory=BattleTrajectory(
                battle_id="live-oracle-108000000", format_id="gen3randombattle", seed=108_000_000
            ),
        )
        config = SimpleNamespace(
            pokezero_player="p1",
            foulplay_player="p2",
            live_continuation_minimum_decision_round=1,
            engine_oracle_belief=False,
            belief_set_source_enabled=lambda: False,
        )
        observation = SimpleNamespace(
            legal_action_mask=tuple(index in {0, 1} for index in range(ACTION_COUNT))
        )

        async def fake_foulplay(**_kwargs: object) -> tuple[str, object]:
            calls.append("decoded-foulplay")
            return "move 2", foulplay_bridge.PolicyDecision(action_index=1, policy_id="foul-play")

        async def fake_to_thread(function: object, /, *args: object, **kwargs: object) -> object:
            return function(*args, **kwargs)  # type: ignore[operator]

        def fake_select(*_args: object, **_kwargs: object) -> object:
            calls.append("source-p1-raw")
            return foulplay_bridge.PolicyDecision(action_index=0, policy_id="raw")

        with (
            patch.object(foulplay_bridge, "_capture_resolved_public_action_round"),
            patch.object(foulplay_bridge, "_player_state", side_effect=lambda *_args, **_kwargs: object()),
            patch.object(foulplay_bridge, "observation_from_player_state", return_value=observation),
            patch.object(foulplay_bridge, "_observation_with_search_metadata", return_value=observation),
            patch.object(
                foulplay_bridge,
                "_context_for_seat",
                return_value=SimpleNamespace(player_id="p1"),
            ),
            patch.object(foulplay_bridge, "_select_live_foulplay_decision", side_effect=fake_foulplay),
            patch.object(foulplay_bridge, "_select_policy_decision", side_effect=fake_select),
            patch.object(foulplay_bridge, "showdown_choice_for_action", return_value="move 2"),
            patch.object(foulplay_bridge.asyncio, "to_thread", side_effect=fake_to_thread),
        ):
            coroutine = foulplay_bridge._handle_decision_boundary(
                config=config,
                bridge=_FakeSourceBridge(calls),
                server=SimpleNamespace(),
                state=state,
                policy=object(),
                vocab=SimpleNamespace(),
                dex=SimpleNamespace(),
                observation_spec=SimpleNamespace(schema_version="v2"),
                decision_round=decision_round,
                requested_players=("p1", "p2"),
                foulplay_process=None,
                foulplay_logs=SimpleNamespace(),
                live_continuation_oracle_controller=_FakeOracleController(calls, failure=failure),
            )
            if failure is None:
                asyncio.run(coroutine)
            else:
                with self.assertRaisesRegex(RuntimeError, "oracle failed"):
                    asyncio.run(coroutine)
        return calls, state

    def _run_boundary_seam(self, *, failure: Exception | None = None) -> list[str]:
        calls: list[str] = []
        state = _ControlledBattleState(
            battle_id="live-118000000",
            seed=118_000_000,
            format_id="gen3randombattle",
            request_lines={"p1": "|request|{}", "p2": "|request|{}"},
            trajectory=BattleTrajectory(
                battle_id="live-118000000", format_id="gen3randombattle", seed=118_000_000
            ),
        )
        config = SimpleNamespace(
            pokezero_player="p1",
            foulplay_player="p2",
            live_continuation_minimum_decision_round=1,
            engine_oracle_belief=False,
            belief_set_source_enabled=lambda: False,
        )
        observation = SimpleNamespace(
            legal_action_mask=tuple(index in {0, 1} for index in range(ACTION_COUNT))
        )

        async def fake_foulplay(**_kwargs: object) -> tuple[str, object]:
            calls.append("decoded-foulplay")
            return "move 2", foulplay_bridge.PolicyDecision(action_index=1, policy_id="foul-play")

        async def fake_to_thread(function: object, /, *args: object, **kwargs: object) -> object:
            return function(*args, **kwargs)  # type: ignore[operator]

        def fake_select(*_args: object, **_kwargs: object) -> object:
            calls.append("source-p1-candidate")
            return foulplay_bridge.PolicyDecision(action_index=0, policy_id="raw")

        with (
            patch.object(foulplay_bridge, "_capture_resolved_public_action_round"),
            patch.object(foulplay_bridge, "_player_state", side_effect=lambda *_args, **_kwargs: object()),
            patch.object(foulplay_bridge, "observation_from_player_state", return_value=observation),
            patch.object(foulplay_bridge, "_observation_with_search_metadata", return_value=observation),
            patch.object(
                foulplay_bridge,
                "_context_for_seat",
                return_value=SimpleNamespace(player_id="p1"),
            ),
            patch.object(foulplay_bridge, "_select_live_foulplay_decision", side_effect=fake_foulplay),
            patch.object(foulplay_bridge, "_select_policy_decision", side_effect=fake_select),
            patch.object(foulplay_bridge, "showdown_choice_for_action", return_value="move 1"),
            patch.object(foulplay_bridge.asyncio, "to_thread", side_effect=fake_to_thread),
        ):
            coroutine = foulplay_bridge._handle_decision_boundary(
                config=config,
                bridge=_FakeSourceBridge(calls),
                server=SimpleNamespace(),
                state=state,
                policy=object(),
                vocab=SimpleNamespace(),
                dex=SimpleNamespace(),
                observation_spec=SimpleNamespace(schema_version="v2"),
                decision_round=1,
                requested_players=("p1", "p2"),
                foulplay_process=None,
                foulplay_logs=SimpleNamespace(),
                live_continuation_probe=_FakeProbe(calls, failure=failure),
            )
            if failure is None:
                asyncio.run(coroutine)
            else:
                with self.assertRaisesRegex(RuntimeError, "continuation failed"):
                    asyncio.run(coroutine)
        return calls

    def test_full_probe_seam_orders_local_proof_before_source_choice_submission(self) -> None:
        self.assertEqual(
            self._run_boundary_seam(),
            [
                "decoded-foulplay",
                "snapshot-bound",
                "source-p1-candidate",
                "local-continuation",
                "source-choices",
            ],
        )

    def test_full_probe_seam_never_submits_source_choices_after_continuation_failure(self) -> None:
        calls = self._run_boundary_seam(failure=RuntimeError("continuation failed"))
        self.assertEqual(
            calls,
            ["decoded-foulplay", "snapshot-bound", "source-p1-candidate", "local-continuation"],
        )

    def test_oracle_controller_binds_external_choice_and_submits_no_raw_fallback(self) -> None:
        calls, state = self._run_oracle_boundary_seam()
        self.assertEqual(
            calls,
            [
                "decoded-foulplay",
                "oracle-snapshot-bound",
                "source-p1-raw",
                "oracle-controller",
                "source-choices",
            ],
        )
        self.assertEqual(len(state.live_continuation_oracle_decisions), 1)
        self.assertEqual(state.trajectory.steps[0].action_index, 1)

    def test_oracle_controller_failure_never_submits_the_raw_source_choice(self) -> None:
        calls, state = self._run_oracle_boundary_seam(failure=RuntimeError("oracle failed"))
        self.assertEqual(
            calls,
            ["decoded-foulplay", "oracle-snapshot-bound", "source-p1-raw", "oracle-controller"],
        )
        self.assertEqual(state.trajectory.steps, [])

    def test_b2_opening_boundary_is_explicitly_counted_raw_before_midgame_oracle_scoring(self) -> None:
        calls, state = self._run_oracle_boundary_seam(decision_round=0)
        self.assertEqual(calls, ["source-p1-raw", "decoded-foulplay", "source-choices"])
        self.assertEqual(state.live_continuation_oracle_decisions, [])
        self.assertEqual(state.live_continuation_oracle_forced_boundary_raw_decisions, 1)

    def test_targeted_snapshot_wait_preserves_unrelated_stream_events(self) -> None:
        async def exercise() -> tuple[object, object]:
            bridge = _BattleBridge(showdown_root=SimpleNamespace(), node_binary="node")
            stream = {"type": "stream", "battleId": "live-118000000", "lines": ["|turn|2"]}
            snapshot = {"type": "snapshot", "battleId": "live-118000000", "snapshot": {}}
            bridge.events.put_nowait(stream)
            bridge.events.put_nowait(snapshot)
            matched = await bridge.next_event_matching(lambda event: event.get("type") == "snapshot")
            deferred = await bridge.next_event()
            return matched, deferred

        matched, deferred = asyncio.run(exercise())
        self.assertEqual(matched["type"], "snapshot")
        self.assertEqual(deferred["type"], "stream")

    def test_restores_fixed_actual_foulplay_action_then_continues(self) -> None:
        env = _FakeEnv()
        snapshot = SimpleNamespace(battle_id="live-118000000", format_id="gen3randombattle")
        boundary = LiveFoulPlayBoundary(
            snapshot=snapshot,
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )
        continuation_calls: list[dict[str, object]] = []

        def fake_continue(**kwargs: object) -> object:
            continuation_calls.append(kwargs)
            return SimpleNamespace(
                decision_round_count=2,
                terminal=TerminalState(winner="p2", turn_count=9, capped=False),
            )

        with patch("pokezero.live_foulplay_continuation.continue_rollout_from_current_state", fake_continue):
            proof = run_live_foulplay_continuation(
                boundary=boundary,
                source_seed=118_000_000,
                source_decision_round=3,
                pokezero_action=4,
                foulplay_action=7,
                foulplay_choice="move 2",
                env_factory=lambda: env,
                continuation_policy_factory=lambda: {"p1": object(), "p2": object()},
                rollout_config=RolloutConfig(max_decision_rounds=10, format_id="gen3randombattle"),
            )

        self.assertEqual(
            env.calls[:3],
            [("reset", 118_000_000, "gen3randombattle"), ("restore", snapshot), ("step", {"p1": 4, "p2": 7})],
        )
        self.assertEqual(proof["decoded_actual_foulplay_action"], 7)
        self.assertEqual(proof["first_restored_joint_step"], {"p1": 4, "p2": 7})
        self.assertEqual(len(continuation_calls), 1)
        self.assertEqual(continuation_calls[0]["starting_decision_round_index"], 4)
        self.assertTrue(continuation_calls[0]["reset_policies"])
        self.assertIs(continuation_calls[0]["available_observations"], env.observations)
        self.assertEqual(env.calls[-1], "close")

    def test_restores_the_correct_seats_when_pokezero_is_p2(self) -> None:
        env = _FakeEnv()
        snapshot = SimpleNamespace(battle_id="live-p2", format_id="gen3randombattle")
        boundary = LiveFoulPlayBoundary(
            snapshot=snapshot,
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )
        with patch(
            "pokezero.live_foulplay_continuation.continue_rollout_from_current_state",
            return_value=SimpleNamespace(
                decision_round_count=1,
                terminal=TerminalState(winner="p2", turn_count=9, capped=False),
            ),
        ):
            proof = run_live_foulplay_continuation(
                boundary=boundary,
                source_seed=108_000_000,
                source_decision_round=1,
                pokezero_action=4,
                foulplay_action=7,
                foulplay_choice="move 2",
                pokezero_player="p2",
                foulplay_player="p1",
                env_factory=lambda: env,
                continuation_policy_factory=lambda: {"p1": object(), "p2": object()},
                rollout_config=RolloutConfig(max_decision_rounds=10, format_id="gen3randombattle"),
            )
        self.assertEqual(env.calls[2], ("step", {"p2": 4, "p1": 7}))
        self.assertEqual(proof["first_restored_joint_step"], {"p2": 4, "p1": 7})

    def test_oracle_scores_all_legal_candidates_without_storing_a_snapshot(self) -> None:
        boundary = LiveFoulPlayBoundary(
            snapshot=SimpleNamespace(battle_id="live-oracle", format_id="gen3randombattle"),
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )
        observed: list[int] = []

        def fake_run(**kwargs: object) -> dict[str, object]:
            action = int(kwargs["pokezero_action"])
            observed.append(action)
            self.assertFalse(kwargs["allow_opening_boundary"])
            self.assertTrue(kwargs["allow_terminal_fixed_step"])
            return {
                "continuation": {
                    "decision_round_count": 2,
                    "terminal_after_fixed_joint_step": False,
                    "terminal": {"winner": "p1" if action == 2 else "p2", "turn_count": 8, "capped": False},
                }
            }

        with patch("pokezero.live_foulplay_continuation.run_live_foulplay_continuation", side_effect=fake_run):
            decision = select_live_foulplay_continuation_oracle_action(
                boundary=boundary,
                source_seed=108_000_000,
                source_decision_round=2,
                raw_action=1,
                legal_actions=(0, 1, 2),
                foulplay_action=3,
                foulplay_choice="move 4",
                pokezero_player="p1",
                foulplay_player="p2",
                candidate_cap=3,
                env_factory=lambda: self.fail("runner is patched"),
                continuation_policy_factory=lambda: self.fail("runner is patched"),
                rollout_config=RolloutConfig(max_decision_rounds=10),
            )
        self.assertEqual(observed, [0, 1, 2])
        self.assertEqual(decision.action_index, 2)
        self.assertEqual(decision.metadata["full_state_snapshot_scope"], "controller-only")
        self.assertNotIn("snapshot", decision.metadata)
        self.assertEqual(decision.metadata["source_request_sha256"], {"p1": "a", "p2": "b"})
        self.assertEqual(decision.metadata["snapshot_request_sha256"], {"p1": "c", "p2": "d"})
        self.assertEqual(decision.metadata["actual_foulplay_choice"], "move 4")
        self.assertEqual(decision.metadata["decoded_actual_foulplay_action"], 3)
        self.assertEqual(decision.metadata["legal_action_indices"], (0, 1, 2))

    def test_oracle_binds_the_per_candidate_eligibility_limit_and_emits_progress(self) -> None:
        boundary = LiveFoulPlayBoundary(
            snapshot=SimpleNamespace(battle_id="live-bounded", format_id="gen3randombattle"),
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )
        progress: list[dict[str, object]] = []

        def fake_run(**kwargs: object) -> dict[str, object]:
            self.assertEqual(kwargs["max_continuation_decision_rounds"], 128)
            action = int(kwargs["pokezero_action"])
            return {
                "continuation": {
                    "decision_round_count": 1,
                    "terminal_after_fixed_joint_step": False,
                    "terminal": {"winner": "p1" if action == 1 else "p2", "turn_count": 4, "capped": False},
                }
            }

        with patch("pokezero.live_foulplay_continuation.run_live_foulplay_continuation", side_effect=fake_run):
            decision = select_live_foulplay_continuation_oracle_action(
                boundary=boundary,
                source_seed=108_000_000,
                source_decision_round=2,
                raw_action=0,
                legal_actions=(0, 1),
                foulplay_action=3,
                foulplay_choice="move 4",
                pokezero_player="p1",
                foulplay_player="p2",
                candidate_cap=2,
                env_factory=lambda: self.fail("runner is patched"),
                continuation_policy_factory=lambda: self.fail("runner is patched"),
                rollout_config=RolloutConfig(max_decision_rounds=100),
                max_continuation_decision_rounds=128,
                progress_callback=lambda event: progress.append(dict(event)),
            )
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.metadata["max_continuation_decision_rounds"], 128)
        self.assertEqual(
            [event["event"] for event in progress],
            [
                "decision-started",
                "candidate-started",
                "candidate-completed",
                "candidate-started",
                "candidate-completed",
                "decision-completed",
            ],
        )

    def test_oracle_expands_only_the_capped_candidate_from_the_same_boundary(self) -> None:
        boundary = LiveFoulPlayBoundary(
            snapshot=SimpleNamespace(battle_id="live-expand", format_id="gen3randombattle"),
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )
        progress: list[dict[str, object]] = []
        calls: list[tuple[int, int]] = []

        def fake_run(**kwargs: object) -> dict[str, object]:
            action = int(kwargs["pokezero_action"])
            bound = int(kwargs["max_continuation_decision_rounds"])
            calls.append((action, bound))
            if action == 0 and bound == 128:
                raise LiveFoulPlayContinuationBoundExceeded("initial cap")
            return {
                "continuation": {
                    "decision_round_count": 143 if action == 0 else 3,
                    "terminal_after_fixed_joint_step": False,
                    "terminal": {"winner": "p1" if action == 0 else "p2", "turn_count": 9, "capped": False},
                }
            }

        with patch("pokezero.live_foulplay_continuation.run_live_foulplay_continuation", side_effect=fake_run):
            decision = select_live_foulplay_continuation_oracle_action(
                boundary=boundary,
                source_seed=108_000_000,
                source_decision_round=2,
                raw_action=1,
                legal_actions=(0, 1),
                foulplay_action=3,
                foulplay_choice="move 4",
                pokezero_player="p1",
                foulplay_player="p2",
                candidate_cap=2,
                env_factory=lambda: self.fail("runner is patched"),
                continuation_policy_factory=lambda: self.fail("runner is patched"),
                rollout_config=RolloutConfig(max_decision_rounds=300),
                max_continuation_decision_rounds=128,
                expanded_continuation_decision_rounds=256,
                progress_callback=lambda event: progress.append(dict(event)),
            )
        self.assertEqual(calls, [(0, 128), (0, 256), (1, 128)])
        self.assertEqual(decision.action_index, 0)
        self.assertEqual(
            [event["event"] for event in progress],
            [
                "decision-started", "candidate-started", "candidate-bound-reached",
                "candidate-expansion-started", "candidate-completed", "candidate-started",
                "candidate-completed", "decision-completed",
            ],
        )
        self.assertEqual(decision.metadata["candidates"][0]["max_continuation_decision_rounds"], 256)

    def test_oracle_records_terminal_fixed_step_candidate(self) -> None:
        boundary = LiveFoulPlayBoundary(
            snapshot=SimpleNamespace(
                battle_id="live-terminal", format_id="gen3randombattle"
            ),
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )

        def fake_run(**kwargs: object) -> dict[str, object]:
            action = int(kwargs["pokezero_action"])
            self.assertTrue(kwargs["allow_terminal_fixed_step"])
            if action == 0:
                return {
                    "continuation": {
                        "decision_round_count": 0,
                        "terminal_after_fixed_joint_step": True,
                        "terminal": {"winner": "p1", "turn_count": 7, "capped": False},
                    }
                }
            return {
                "continuation": {
                    "decision_round_count": 1,
                    "terminal_after_fixed_joint_step": False,
                    "terminal": {"winner": "p2", "turn_count": 8, "capped": False},
                }
            }

        with patch(
            "pokezero.live_foulplay_continuation.run_live_foulplay_continuation",
            side_effect=fake_run,
        ):
            decision = select_live_foulplay_continuation_oracle_action(
                boundary=boundary,
                source_seed=108_000_000,
                source_decision_round=2,
                raw_action=1,
                legal_actions=(0, 1),
                foulplay_action=3,
                foulplay_choice="move 4",
                pokezero_player="p1",
                foulplay_player="p2",
                candidate_cap=2,
                env_factory=lambda: self.fail("runner is patched"),
                continuation_policy_factory=lambda: self.fail("runner is patched"),
                rollout_config=RolloutConfig(max_decision_rounds=10),
            )

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(
            decision.metadata["candidates"],
            (
                {
                    "action_index": 0,
                    "score": 1.0,
                    "continuation_decision_round_count": 0,
                    "terminal": {"winner": "p1", "turn_count": 7, "capped": False},
                    "terminal_after_fixed_joint_step": True,
                },
                {
                    "action_index": 1,
                    "score": 0.0,
                    "continuation_decision_round_count": 1,
                    "terminal": {"winner": "p2", "turn_count": 8, "capped": False},
                    "terminal_after_fixed_joint_step": False,
                },
            ),
        )

    def test_oracle_handles_opening_boundary_and_terminal_fixed_joint_step(self) -> None:
        env = _FakeTerminalEnv()
        boundary = LiveFoulPlayBoundary(
            snapshot=SimpleNamespace(battle_id="opening", format_id="gen3randombattle"),
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )
        proof = run_live_foulplay_continuation(
            boundary=boundary,
            source_seed=108_000_000,
            source_decision_round=0,
            pokezero_action=1,
            foulplay_action=2,
            foulplay_choice="move 3",
            allow_opening_boundary=True,
            allow_terminal_fixed_step=True,
            env_factory=lambda: env,
            continuation_policy_factory=lambda: self.fail("terminal fixed step needs no continuation policies"),
            rollout_config=RolloutConfig(max_decision_rounds=2),
        )
        self.assertEqual(proof["continuation"]["decision_round_count"], 0)
        self.assertTrue(proof["continuation"]["terminal_after_fixed_joint_step"])
        self.assertEqual(proof["continuation"]["terminal"]["winner"], "p2")

    def test_candidate_continuation_bound_is_fail_closed(self) -> None:
        env = _FakeEnv()
        boundary = LiveFoulPlayBoundary(
            snapshot=SimpleNamespace(battle_id="bounded", format_id="gen3randombattle"),
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )
        capped = SimpleNamespace(
            terminal=TerminalState(winner=None, turn_count=9, capped=True),
            decision_round_count=5,
        )
        with patch(
            "pokezero.live_foulplay_continuation.continue_rollout_from_current_state",
            return_value=capped,
        ) as continued:
            with self.assertRaisesRegex(LiveFoulPlayContinuationError, "5-decision eligibility bound"):
                run_live_foulplay_continuation(
                    boundary=boundary,
                    source_seed=108_000_000,
                    source_decision_round=2,
                    pokezero_action=1,
                    foulplay_action=2,
                    foulplay_choice="move 3",
                    env_factory=lambda: env,
                    continuation_policy_factory=lambda: {"p1": object(), "p2": object()},
                    rollout_config=RolloutConfig(max_decision_rounds=1024),
                    max_continuation_decision_rounds=5,
                )
        self.assertEqual(continued.call_args.kwargs["config"].max_decision_rounds, 8)
        self.assertEqual(env.calls[-1], "close")

    def test_oracle_receipt_is_seat_relative_for_p2(self) -> None:
        boundary = LiveFoulPlayBoundary(
            snapshot=SimpleNamespace(battle_id="live-oracle-p2", format_id="gen3randombattle"),
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )

        def fake_run(**kwargs: object) -> dict[str, object]:
            self.assertEqual(kwargs["pokezero_player"], "p2")
            self.assertEqual(kwargs["foulplay_player"], "p1")
            action = int(kwargs["pokezero_action"])
            return {
                "continuation": {
                    "decision_round_count": 1,
                    "terminal_after_fixed_joint_step": False,
                    "terminal": {"winner": "p2" if action == 1 else "p1", "turn_count": 8, "capped": False},
                }
            }

        with patch("pokezero.live_foulplay_continuation.run_live_foulplay_continuation", side_effect=fake_run):
            decision = select_live_foulplay_continuation_oracle_action(
                boundary=boundary,
                source_seed=108_000_001,
                source_decision_round=1,
                raw_action=0,
                legal_actions=(0, 1),
                foulplay_action=3,
                foulplay_choice="move 4",
                pokezero_player="p2",
                foulplay_player="p1",
                candidate_cap=2,
                env_factory=lambda: self.fail("runner is patched"),
                continuation_policy_factory=lambda: self.fail("runner is patched"),
                rollout_config=RolloutConfig(max_decision_rounds=10),
            )
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.metadata["first_restored_joint_step"], {"p2": 1, "p1": 3})

    def test_b2_oracle_refuses_opening_round_candidates(self) -> None:
        with self.assertRaisesRegex(LiveFoulPlayContinuationError, "mid-game source decision round"):
            select_live_foulplay_continuation_oracle_action(
                boundary=SimpleNamespace(),
                source_seed=108_000_000,
                source_decision_round=0,
                raw_action=0,
                legal_actions=(0,),
                foulplay_action=1,
                foulplay_choice="move 2",
                pokezero_player="p1",
                foulplay_player="p2",
                candidate_cap=1,
                env_factory=lambda: self.fail("opening must fail before a restore"),
                continuation_policy_factory=lambda: self.fail("opening must fail before a restore"),
                rollout_config=RolloutConfig(max_decision_rounds=10),
            )

    def test_oracle_refuses_to_truncate_a_live_legal_candidate_set(self) -> None:
        with self.assertRaisesRegex(LiveFoulPlayContinuationError, "refusing to truncate"):
            select_live_foulplay_continuation_oracle_action(
                boundary=SimpleNamespace(),  # not reached before the cap refusal
                source_seed=1,
                source_decision_round=1,
                raw_action=0,
                legal_actions=(0, 1),
                foulplay_action=0,
                foulplay_choice="move 1",
                pokezero_player="p1",
                foulplay_player="p2",
                candidate_cap=1,
                env_factory=lambda: self.fail("must not construct env"),
                continuation_policy_factory=lambda: self.fail("must not construct policies"),
                rollout_config=RolloutConfig(max_decision_rounds=10),
            )

    def test_rejects_corrupt_snapshot_request_before_any_restore_or_continuation(self) -> None:
        with self.assertRaisesRegex(
            LiveFoulPlayContinuationError,
            "snapshot p2 boundary request does not match",
        ):
            reconstruct_live_foulplay_boundary(
                source_battle_id="live-118000000",
                format_id="gen3randombattle",
                bridge_snapshot={
                    "battle": {"opaque": True},
                    "boundaryRequests": {"p1": {"rqid": 1}, "p2": {"rqid": 999}},
                },
                public_protocol_lines=("|turn|2",),
                current_request_lines={
                    "p1": "|request|{\"rqid\":1}",
                    "p2": "|request|{\"rqid\":2}",
                },
                request_history_lines={"p1": (), "p2": ()},
                belief_set_source=None,
            )

    def test_reconstructs_both_live_request_histories_for_the_restored_shell(self) -> None:
        boundary = reconstruct_live_foulplay_boundary(
            source_battle_id="live-118000000",
            format_id="gen3randombattle",
            bridge_snapshot={
                "battle": {"opaque": True},
                "boundaryRequests": {"p1": {"rqid": 3}, "p2": {"rqid": 4}},
            },
            public_protocol_lines=("|turn|2",),
            current_request_lines={
                "p1": "|request|{\"rqid\":3}",
                "p2": "|request|{\"rqid\":4}",
            },
            request_history_lines={
                "p1": ("|request|{\"rqid\":1}", "|request|{\"rqid\":3}"),
                "p2": ("|request|{\"rqid\":2}", "|request|{\"rqid\":4}"),
            },
            belief_set_source=None,
        )
        self.assertEqual(boundary.snapshot.latest_turn, 2)
        self.assertEqual(boundary.snapshot.first_requests["p1"], {"rqid": 1})
        self.assertEqual(boundary.snapshot.latest_requests["p2"], {"rqid": 4})
        self.assertEqual(len(boundary.snapshot.request_history["p1"]), 2)

    def test_rejects_opening_round_before_creating_a_local_environment(self) -> None:
        boundary = LiveFoulPlayBoundary(
            snapshot=SimpleNamespace(battle_id="live-118000000", format_id="gen3randombattle"),
            source_request_sha256={"p1": "a", "p2": "b"},
            snapshot_request_sha256={"p1": "c", "p2": "d"},
        )
        with self.assertRaisesRegex(LiveFoulPlayContinuationError, "opening-round"):
            run_live_foulplay_continuation(
                boundary=boundary,
                source_seed=118_000_000,
                source_decision_round=0,
                pokezero_action=4,
                foulplay_action=7,
                foulplay_choice="move 2",
                env_factory=lambda: self.fail("must not create an environment for round zero"),
                continuation_policy_factory=lambda: self.fail("must not create policies for round zero"),
                rollout_config=RolloutConfig(max_decision_rounds=10, format_id="gen3randombattle"),
            )


if __name__ == "__main__":
    unittest.main()
