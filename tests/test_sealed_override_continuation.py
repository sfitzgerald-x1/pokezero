from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.env import StepResult, TerminalState
from pokezero.rollout import RolloutConfig
from pokezero.sealed_override_continuation import (
    SEALED_OVERRIDE_CONTINUATION_SCHEMA_VERSION,
    SealedOverrideContinuationCapError,
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
        environments = [_FakeEnv() for _ in range(12)]
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
                    "policy_consistent": policy_factory("policy_consistent"),
                    "uniform_own": policy_factory("uniform_own"),
                },
                continuation_rng_seeds=[101, 102],
                search_evidence={"root_q_gap": 0.125},
                env_factory=env_factory,
                rollout_config=RolloutConfig(max_decision_rounds=100),
            )

        self.assertEqual(readout["schema_version"], "pokezero.sealed-root-action-grid.v2")
        self.assertEqual(len(factories), 12)
        self.assertEqual(
            [target["target"] for target in readout["continuation_targets"]],
            ["policy_consistent", "uniform_own"],
        )
        for target in readout["continuation_targets"]:
            self.assertEqual([trial["continuation_rng_seed"] for trial in target["trials"]], [101, 102])
            for trial in target["trials"]:
                self.assertEqual(
                    [outcome["action_index"] for outcome in trial["outcomes"]], [2, 4, 7]
                )
                self.assertNotIn("opponent_action", trial)
                self.assertNotIn("snapshot", trial)
        self.assertNotIn("opponent_action", readout)
        self.assertNotIn("snapshot", readout)

    def test_root_action_grid_retries_only_a_capped_suffix_with_registered_bound(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run(**kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            if len(calls) == 1:
                raise SealedOverrideContinuationCapError("independent continuation capped before a terminal result")
            decision_round_count = 4 if len(calls) == 2 else 3
            return {
                "continuation": {
                    "decision_round_count": decision_round_count,
                    "terminal_after_fixed_joint_step": False,
                    "terminal": {"winner": "p1", "turn_count": 10, "capped": False},
                }
            }

        with patch(
            "pokezero.sealed_override_continuation.run_sealed_override_continuation",
            fake_run,
        ):
            readout = evaluate_sealed_root_action_grid(
                snapshot=self._snapshot(),
                source_battle_id="source-grid-retry",
                source_seed=20_260_923,
                source_decision_round=6,
                subject_player="p1",
                actions={"raw_policy": 2, "mcts_selected": 4},
                opponent_player="p2",
                opponent_action=1,
                continuation_policy_factories={"policy_consistent": lambda: {}},
                continuation_rng_seeds=[101],
                search_evidence={},
                env_factory=lambda: self.fail("patched runner must not allocate an environment"),
                rollout_config=RolloutConfig(max_decision_rounds=100),
                max_continuation_decision_rounds=3,
                expanded_max_continuation_decision_rounds=11,
            )

        self.assertEqual(
            [call["max_continuation_decision_rounds"] for call in calls], [3, 11, 3]
        )
        self.assertTrue(all(call["source_seed"] == 20_260_923 for call in calls))
        self.assertTrue(all(call["continuation_rng_seed"] == 101 for call in calls))
        outcomes = readout["continuation_targets"][0]["trials"][0]["outcomes"]
        self.assertEqual(outcomes[0]["continuation"]["cap_retry"], True)
        self.assertEqual(
            outcomes[0]["continuation"]["effective_max_continuation_decision_rounds"], 11
        )
        self.assertEqual(outcomes[1]["continuation"]["cap_retry"], False)
        self.assertEqual(
            outcomes[1]["continuation"]["effective_max_continuation_decision_rounds"], 3
        )

    def test_root_action_grid_refuses_non_increasing_retry_bound(self) -> None:
        with self.assertRaisesRegex(SealedOverrideContinuationError, "must exceed"):
            evaluate_sealed_root_action_grid(
                snapshot=self._snapshot(),
                source_battle_id="bad-grid-retry",
                source_seed=1,
                source_decision_round=1,
                subject_player="p1",
                actions={"raw_policy": 2, "mcts_selected": 4},
                opponent_player="p2",
                opponent_action=1,
                continuation_policy_factories={"policy_consistent": lambda: {}},
                continuation_rng_seeds=[1],
                search_evidence={},
                env_factory=lambda: self.fail("must not allocate environment"),
                rollout_config=RolloutConfig(),
                max_continuation_decision_rounds=10,
                expanded_max_continuation_decision_rounds=10,
            )

    def test_root_action_grid_fails_closed_when_expanded_retry_also_caps(self) -> None:
        calls: list[dict[str, object]] = []

        def always_cap(**kwargs: object) -> object:
            calls.append(dict(kwargs))
            raise SealedOverrideContinuationCapError("independent continuation capped before a terminal result")

        with patch(
            "pokezero.sealed_override_continuation.run_sealed_override_continuation",
            always_cap,
        ), self.assertRaisesRegex(SealedOverrideContinuationCapError, "capped before a terminal"):
            evaluate_sealed_root_action_grid(
                snapshot=self._snapshot(),
                source_battle_id="double-cap-grid",
                source_seed=1,
                source_decision_round=1,
                subject_player="p1",
                actions={"raw_policy": 2, "mcts_selected": 4},
                opponent_player="p2",
                opponent_action=1,
                continuation_policy_factories={"policy_consistent": lambda: {}},
                continuation_rng_seeds=[1],
                search_evidence={},
                env_factory=lambda: self.fail("patched runner must not allocate an environment"),
                rollout_config=RolloutConfig(),
                max_continuation_decision_rounds=3,
                expanded_max_continuation_decision_rounds=11,
            )

        self.assertEqual(
            [call["max_continuation_decision_rounds"] for call in calls], [3, 11]
        )

    def test_root_action_grid_rejects_a_retry_that_does_not_pass_the_original_ceiling(self) -> None:
        calls: list[dict[str, object]] = []

        def cap_then_inconsistent_terminal(**kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            if len(calls) == 1:
                raise SealedOverrideContinuationCapError("independent continuation capped before a terminal result")
            return {
                "continuation": {
                    "decision_round_count": 3,
                    "terminal_after_fixed_joint_step": False,
                    "terminal": {"winner": "p1", "turn_count": 10, "capped": False},
                }
            }

        with patch(
            "pokezero.sealed_override_continuation.run_sealed_override_continuation",
            cap_then_inconsistent_terminal,
        ), self.assertRaisesRegex(SealedOverrideContinuationError, "did not run beyond"):
            evaluate_sealed_root_action_grid(
                snapshot=self._snapshot(),
                source_battle_id="inconsistent-retry-grid",
                source_seed=1,
                source_decision_round=1,
                subject_player="p1",
                actions={"raw_policy": 2, "mcts_selected": 4},
                opponent_player="p2",
                opponent_action=1,
                continuation_policy_factories={"policy_consistent": lambda: {}},
                continuation_rng_seeds=[1],
                search_evidence={},
                env_factory=lambda: self.fail("patched runner must not allocate an environment"),
                rollout_config=RolloutConfig(),
                max_continuation_decision_rounds=3,
                expanded_max_continuation_decision_rounds=11,
            )

        self.assertEqual(
            [call["max_continuation_decision_rounds"] for call in calls], [3, 11]
        )

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
