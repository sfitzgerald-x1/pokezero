from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.env import StepResult, TerminalState
from pokezero.rollout import RolloutConfig
from pokezero.sealed_override_continuation import (
    SEALED_OVERRIDE_CONTINUATION_SCHEMA_VERSION,
    SealedOverrideContinuationError,
    evaluate_sealed_override_pair,
    evaluate_sealed_root_action_grid,
    run_sealed_override_continuation,
)


class _FakeEnv:
    def __init__(self, *, terminal_after_step: TerminalState | None = None) -> None:
        self.terminal_after_step = terminal_after_step
        self.calls: list[object] = []

    def reset(self, *, seed: int, format_id: str) -> None:
        self.calls.append(("reset", seed, format_id))

    def restore(self, snapshot: object) -> None:
        self.calls.append(("restore", snapshot))

    def requested_players(self) -> tuple[str, str]:
        return ("p1", "p2")

    def terminal(self) -> None:
        return None

    def step(self, actions: dict[str, int]) -> StepResult:
        self.calls.append(("step", dict(actions)))
        return StepResult(
            observations={"p1": object(), "p2": object()},
            rewards={},
            terminal=self.terminal_after_step,
        )

    def close(self) -> None:
        self.calls.append("close")


class SealedOverrideContinuationTest(unittest.TestCase):
    def _snapshot(self) -> object:
        # The executor deliberately uses only these public shell identifiers;
        # a real LocalShowdownSnapshot is type-checked in the production path.
        return SimpleNamespace(battle_id="audit-17", format_id="gen3randombattle")

    def test_restores_fixed_joint_action_then_returns_only_terminal_summary(self) -> None:
        env = _FakeEnv()
        continuation_calls: list[dict[str, object]] = []
        def source_sink(_: object) -> None:
            return None

        def fake_continue(**kwargs: object) -> object:
            continuation_calls.append(kwargs)
            return SimpleNamespace(
                decision_round_count=3,
                terminal=TerminalState(winner="p1", turn_count=9, capped=False),
            )

        with patch(
            "pokezero.sealed_override_continuation.LocalShowdownSnapshot",
            SimpleNamespace,
        ), patch(
            "pokezero.sealed_override_continuation.continue_rollout_from_current_state",
            fake_continue,
        ):
            readout = run_sealed_override_continuation(
                snapshot=self._snapshot(),
                source_battle_id="source-rollout-17",
                source_seed=20_260_920,
                source_decision_round=4,
                subject_player="p1",
                subject_action=4,
                opponent_player="p2",
                opponent_action=7,
                action_label="mcts-override",
                env_factory=lambda: env,
                continuation_policy_factory=lambda: {"p1": object(), "p2": object()},
                rollout_config=RolloutConfig(
                    max_decision_rounds=100,
                    decision_sink=source_sink,
                    public_decision_sink=source_sink,
                    sealed_pre_step_sink=source_sink,
                ),
                max_continuation_decision_rounds=25,
            )

        self.assertEqual(
            env.calls[:3],
            [
                ("reset", 20_260_920, "gen3randombattle"),
                ("restore", self._snapshot()),
                ("step", {"p1": 4, "p2": 7}),
            ],
        )
        self.assertEqual(readout["schema_version"], SEALED_OVERRIDE_CONTINUATION_SCHEMA_VERSION)
        self.assertEqual(readout["source_battle_id"], "source-rollout-17")
        self.assertEqual(readout["subject_action"], 4)
        self.assertTrue(readout["opponent_action_held_fixed"])
        self.assertNotIn("fixed_joint_action", readout)
        self.assertNotIn("snapshot", readout)
        self.assertEqual(readout["continuation"]["terminal"]["winner"], "p1")
        self.assertFalse(readout["continuation"]["terminal_after_fixed_joint_step"])
        self.assertEqual(continuation_calls[0]["starting_decision_round_index"], 5)
        self.assertFalse(continuation_calls[0]["reset_policies"])
        self.assertEqual(continuation_calls[0]["config"].max_decision_rounds, 30)
        self.assertIsNone(continuation_calls[0]["config"].decision_sink)
        self.assertIsNone(continuation_calls[0]["config"].public_decision_sink)
        self.assertIsNone(continuation_calls[0]["config"].sealed_pre_step_sink)
        self.assertEqual(env.calls[-1], "close")

    def test_records_immediate_terminal_as_valid_independent_evidence(self) -> None:
        env = _FakeEnv(terminal_after_step=TerminalState(winner="p2", turn_count=5, capped=False))
        with patch(
            "pokezero.sealed_override_continuation.LocalShowdownSnapshot",
            SimpleNamespace,
        ), patch(
            "pokezero.sealed_override_continuation.continue_rollout_from_current_state",
        ) as continuation:
            readout = run_sealed_override_continuation(
                snapshot=self._snapshot(),
                source_battle_id="source-rollout-immediate",
                source_seed=20_260_921,
                source_decision_round=2,
                subject_player="p2",
                subject_action=3,
                opponent_player="p1",
                opponent_action=1,
                action_label="raw-policy",
                env_factory=lambda: env,
                continuation_policy_factory=lambda: self.fail("must not create policies"),
                rollout_config=RolloutConfig(max_decision_rounds=100),
            )

        continuation.assert_not_called()
        self.assertTrue(readout["continuation"]["terminal_after_fixed_joint_step"])
        self.assertEqual(readout["continuation"]["decision_round_count"], 0)
        self.assertEqual(readout["continuation"]["terminal"]["winner"], "p2")

    def test_rejects_non_local_snapshot_before_creating_environment(self) -> None:
        with self.assertRaisesRegex(SealedOverrideContinuationError, "LocalShowdownSnapshot"):
            run_sealed_override_continuation(
                snapshot=object(),
                source_battle_id="source-rollout-invalid",
                source_seed=1,
                source_decision_round=1,
                subject_player="p1",
                subject_action=0,
                opponent_player="p2",
                opponent_action=0,
                action_label="raw-policy",
                env_factory=lambda: self.fail("must not allocate environment"),
                continuation_policy_factory=lambda: self.fail("must not allocate policies"),
                rollout_config=RolloutConfig(),
            )

    def test_pair_uses_fresh_environment_and_policies_for_both_actions(self) -> None:
        environments = [_FakeEnv(), _FakeEnv()]
        policy_factory_calls = 0

        def env_factory() -> _FakeEnv:
            return environments.pop(0)

        def policy_factory() -> dict[str, object]:
            nonlocal policy_factory_calls
            policy_factory_calls += 1
            return {"p1": object(), "p2": object()}

        def fake_continue(**kwargs: object) -> object:
            action = kwargs["env"].calls[-1][1]["p1"]
            return SimpleNamespace(
                decision_round_count=action,
                terminal=TerminalState(winner="p1", turn_count=10 + action, capped=False),
            )

        with patch(
            "pokezero.sealed_override_continuation.LocalShowdownSnapshot",
            SimpleNamespace,
        ), patch(
            "pokezero.sealed_override_continuation.continue_rollout_from_current_state",
            fake_continue,
        ):
            readout = evaluate_sealed_override_pair(
                snapshot=self._snapshot(),
                source_battle_id="source-rollout-pair",
                source_seed=20_260_922,
                source_decision_round=3,
                subject_player="p1",
                mcts_action=4,
                raw_action=2,
                opponent_player="p2",
                opponent_action=7,
                search_evidence={"root_q_gap": 0.125, "root_visit_gap": 0.25},
                env_factory=env_factory,
                continuation_policy_factory=policy_factory,
                rollout_config=RolloutConfig(max_decision_rounds=100),
            )

        self.assertEqual(policy_factory_calls, 2)
        self.assertEqual(readout["mcts_action"], 4)
        self.assertEqual(readout["source_battle_id"], "source-rollout-pair")
        self.assertEqual(readout["raw_action"], 2)
        self.assertTrue(readout["opponent_action_held_fixed"])
        self.assertNotIn("opponent_action", readout)
        self.assertNotIn("opponent_action", readout["mcts"])
        self.assertEqual(readout["mcts"]["decision_round_count"], 4)
        self.assertEqual(readout["raw"]["decision_round_count"], 2)
        self.assertEqual(readout["search_evidence"], {"root_q_gap": 0.125, "root_visit_gap": 0.25})
        self.assertNotIn("snapshot", readout)

    def test_pair_refuses_a_non_override(self) -> None:
        with self.assertRaisesRegex(SealedOverrideContinuationError, "distinct MCTS and raw"):
            evaluate_sealed_override_pair(
                snapshot=self._snapshot(),
                source_battle_id="source-rollout-nonoverride",
                source_seed=1,
                source_decision_round=1,
                subject_player="p1",
                mcts_action=1,
                raw_action=1,
                opponent_player="p2",
                opponent_action=2,
                search_evidence={},
                env_factory=lambda: self.fail("must not allocate environment"),
                continuation_policy_factory=lambda: self.fail("must not allocate policies"),
                rollout_config=RolloutConfig(),
            )

    def test_root_action_grid_pairs_every_action_within_each_target_and_trial(self) -> None:
        environments = [_FakeEnv() for _ in range(18)]
        factories: list[str] = []

        def env_factory() -> _FakeEnv:
            return environments.pop(0)

        def policy_factory(mode: str):
            def factory() -> dict[str, object]:
                factories.append(mode)
                return {"p1": object(), "p2": object()}
            return factory

        def fake_continue(**kwargs: object) -> object:
            action = kwargs["env"].calls[-1][1]["p1"]
            return SimpleNamespace(
                decision_round_count=action,
                terminal=TerminalState(winner="p1", turn_count=10 + action, capped=False),
            )

        with patch(
            "pokezero.sealed_override_continuation.LocalShowdownSnapshot",
            SimpleNamespace,
        ), patch(
            "pokezero.sealed_override_continuation.continue_rollout_from_current_state",
            fake_continue,
        ):
            readout = evaluate_sealed_root_action_grid(
                snapshot=self._snapshot(),
                source_battle_id="source-grid",
                source_seed=20_260_923,
                source_decision_round=6,
                subject_player="p1",
                actions={"raw_policy": 2, "mcts_selected": 4, "visit_alternative": 7},
                opponent_player="p2",
                opponent_action=1,
                continuation_policy_factories={
                    "deployed_raw": policy_factory("deployed_raw"),
                    "policy_consistent": policy_factory("policy_consistent"),
                    "uniform_own": policy_factory("uniform_own"),
                },
                continuation_rng_seeds={
                    "deployed_raw": [101],
                    "policy_consistent": [101, 102],
                    "uniform_own": [101, 102],
                },
                search_evidence={"root_q_gap": 0.125},
                env_factory=env_factory,
                rollout_config=RolloutConfig(max_decision_rounds=100),
            )

        self.assertEqual(readout["schema_version"], "pokezero.sealed-root-action-grid.v1")
        self.assertEqual(len(factories), 15)
        self.assertEqual(
            [target["target"] for target in readout["continuation_targets"]],
            ["deployed_raw", "policy_consistent", "uniform_own"],
        )
        for target in readout["continuation_targets"]:
            expected_seeds = [101] if target["target"] == "deployed_raw" else [101, 102]
            self.assertEqual(
                [trial["continuation_rng_seed"] for trial in target["trials"]],
                expected_seeds,
            )
            for trial in target["trials"]:
                self.assertEqual(
                    [outcome["action_index"] for outcome in trial["outcomes"]], [2, 4, 7]
                )
                self.assertNotIn("opponent_action", trial)
                self.assertNotIn("snapshot", trial)
        self.assertNotIn("opponent_action", readout)
        self.assertNotIn("snapshot", readout)

    def test_root_action_grid_rejects_repeated_actions_before_allocating(self) -> None:
        with self.assertRaisesRegex(SealedOverrideContinuationError, "repeats an action"):
            evaluate_sealed_root_action_grid(
                snapshot=self._snapshot(),
                source_battle_id="bad-grid",
                source_seed=1,
                source_decision_round=1,
                subject_player="p1",
                actions={"raw_policy": 2, "mcts_selected": 2},
                opponent_player="p2",
                opponent_action=1,
                continuation_policy_factories={"policy_consistent": lambda: {}},
                continuation_rng_seeds=[1],
                search_evidence={},
                env_factory=lambda: self.fail("must not allocate environment"),
                rollout_config=RolloutConfig(),
            )

    def test_root_action_grid_rejects_a_target_schedule_that_does_not_match_factories(self) -> None:
        with self.assertRaisesRegex(SealedOverrideContinuationError, "schedules do not match"):
            evaluate_sealed_root_action_grid(
                snapshot=self._snapshot(),
                source_battle_id="bad-schedule",
                source_seed=1,
                source_decision_round=1,
                subject_player="p1",
                actions={"raw_policy": 2, "mcts_selected": 4},
                opponent_player="p2",
                opponent_action=1,
                continuation_policy_factories={"deployed_raw": lambda: {}},
                continuation_rng_seeds={"wrong-target": [1]},
                search_evidence={},
                env_factory=lambda: self.fail("must not allocate environment"),
                rollout_config=RolloutConfig(),
            )
