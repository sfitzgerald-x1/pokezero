from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.env import StepResult, TerminalState
from pokezero import foulplay_bridge
from pokezero.foulplay_bridge import _BattleBridge, _ControlledBattleState
from pokezero.live_foulplay_continuation import (
    LiveFoulPlayBoundary,
    LiveFoulPlayContinuationError,
    reconstruct_live_foulplay_boundary,
    run_live_foulplay_continuation,
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


class LiveFoulPlayContinuationTest(unittest.TestCase):
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
