import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import (
    CollectionMetrics,
    collect_rollouts,
    policy_from_name,
    read_rollout_records,
    rollout_record_from_dict,
    rollout_record_to_dict,
    summarize_records,
)
from pokezero.env import StepResult, TerminalState
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig, LocalShowdownEnv
from pokezero.observation import ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.policy import RandomLegalPolicy
from pokezero.rollout import RolloutConfig
from pokezero.rollout_cli import main as rollout_cli_main
from pokezero.trajectory import BattleTrajectory, TrajectoryStep, trajectory_from_dict, trajectory_to_dict


def observation(mask: tuple[bool, ...]) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((index,) for index in range(spec.token_count)),
        numeric_features=tuple((float(index),) for index in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=mask,
        perspective=ObservationPerspective.from_showdown_slot("p1", "p1"),
    )


def trajectory() -> BattleTrajectory:
    mask = (True, False, False, False, False, False, False, False, False)
    result = BattleTrajectory(
        battle_id="battle-1",
        format_id="gen3randombattle",
        seed=123,
        metadata={"max_decision_rounds": 250},
    )
    result.append(
        TrajectoryStep(
            player_id="p1",
            turn_index=0,
            observation=observation(mask),
            legal_action_mask=mask,
            action_index=0,
            reward=1.0,
            opponent_action_index=1,
            action_probability=0.5,
            metadata={"policy_id": "random-legal"},
        )
    )
    result.record_terminal(TerminalState(winner="p1", turn_count=12))
    return result


class OneTurnEnv:
    def __init__(self) -> None:
        self._observation = observation((True, False, False, False, False, False, False, False, False))
        self._requested = ("p1", "p2")
        self._terminal = None
        self.closed = False

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self._requested = ("p1", "p2")
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return self._observation

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self._observation.legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return self._requested

    def step(self, actions: dict[str, int]) -> StepResult:
        self._requested = ()
        self._terminal = TerminalState(winner="p1", turn_count=1)
        return StepResult(
            observations={},
            rewards={"p1": 1.0, "p2": -1.0},
            terminal=self._terminal,
            requested_players=(),
        )

    def terminal(self) -> TerminalState | None:
        return self._terminal

    def close(self) -> None:
        self.closed = True


def integration_config() -> LocalShowdownConfig | None:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=10.0)


class CollectionTest(unittest.TestCase):
    def test_trajectory_dict_round_trip_preserves_observation_and_terminal(self) -> None:
        original = trajectory()

        restored = trajectory_from_dict(trajectory_to_dict(original))

        self.assertEqual(restored.battle_id, original.battle_id)
        self.assertEqual(restored.terminal, original.terminal)
        self.assertEqual(restored.steps[0].observation.perspective.showdown_slot, "p1")
        self.assertEqual(restored.steps[0].action_probability, 0.5)

    def test_rollout_record_dict_round_trip(self) -> None:
        metrics = summarize_records([], elapsed_seconds=1.0)
        self.assertEqual(metrics.games, 0)
        record = collect_one_record_for_test()

        restored = rollout_record_from_dict(rollout_record_to_dict(record))

        self.assertEqual(restored.battle_id, record.battle_id)
        self.assertEqual(restored.policy_ids, record.policy_ids)
        self.assertEqual(restored.terminal, TerminalState(winner="p1", turn_count=1))
        self.assertEqual(len(restored.trajectory.steps), 2)

    def test_collect_rollouts_writes_jsonl_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"

            metrics = collect_rollouts(
                output_path=output_path,
                games=2,
                env_factory=OneTurnEnv,
                policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=10,
            )

            records = read_rollout_records(output_path)
        self.assertEqual(metrics.games, 2)
        self.assertEqual(metrics.p1_wins, 2)
        self.assertEqual(metrics.total_decision_rounds, 2)
        self.assertEqual([record.seed for record in records], [10, 11])
        self.assertEqual([record.battle_id for record in records], ["rollout-10", "rollout-11"])

    def test_policy_from_name_rejects_unknown_policy(self) -> None:
        self.assertEqual(policy_from_name("random-legal").policy_id, "random-legal")
        self.assertEqual(policy_from_name("simple-legal").policy_id, "simple-legal")
        with self.assertRaisesRegex(ValueError, "Unsupported policy"):
            policy_from_name("unknown")

    def test_rollout_cli_collect_wires_arguments_and_prints_metrics(self) -> None:
        fake_metrics = CollectionMetrics(
            games=1,
            elapsed_seconds=2.0,
            total_decision_rounds=4,
            total_simulator_turns=3,
            p1_wins=1,
            p2_wins=0,
            ties=0,
            capped_games=0,
        )
        with patch("pokezero.rollout_cli.collect_rollouts", return_value=fake_metrics) as collect:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = rollout_cli_main(
                    [
                        "collect",
                        "--games",
                        "1",
                        "--out",
                        "runs/test.jsonl",
                        "--seed-start",
                        "50",
                        "--max-decision-rounds",
                        "7",
                        "--p1-policy",
                        "simple-legal",
                        "--p2-policy",
                        "random-legal",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = collect.call_args.kwargs
        self.assertEqual(kwargs["games"], 1)
        self.assertEqual(kwargs["seed_start"], 50)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 7)
        self.assertEqual(kwargs["policies"]["p1"].policy_id, "simple-legal")
        self.assertIn("games_per_second: 0.500", stdout.getvalue())

    @unittest.skipIf(integration_config() is None, "requires node and built Pokemon Showdown checkout")
    def test_collect_rollouts_smoke_with_local_showdown_env(self) -> None:
        config = integration_config()
        assert config is not None
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "showdown.jsonl"

            metrics = collect_rollouts(
                output_path=output_path,
                games=1,
                env_factory=lambda: LocalShowdownEnv(config),
                policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=30),
                seed_start=3,
            )

            records = read_rollout_records(output_path)
        self.assertEqual(metrics.games, 1)
        self.assertEqual(len(records), 1)
        self.assertGreater(len(records[0].trajectory.steps), 0)
        self.assertIn(records[0].terminal.winner, {"p1", "p2", None})


def collect_one_record_for_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "rollouts.jsonl"
        collect_rollouts(
            output_path=output_path,
            games=1,
            env_factory=OneTurnEnv,
            policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
            rollout_config=RolloutConfig(max_decision_rounds=5),
        )
        return read_rollout_records(output_path)[0]


if __name__ == "__main__":
    unittest.main()
