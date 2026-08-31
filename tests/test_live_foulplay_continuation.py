from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.env import StepResult, TerminalState
from pokezero.foulplay_bridge import _BattleBridge
from pokezero.live_foulplay_continuation import (
    LiveFoulPlayBoundary,
    LiveFoulPlayContinuationError,
    reconstruct_live_foulplay_boundary,
    run_live_foulplay_continuation,
)
from pokezero.rollout import RolloutConfig


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


class LiveFoulPlayContinuationTest(unittest.TestCase):
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
