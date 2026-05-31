import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import (
    BenchmarkMatchup,
    BenchmarkMatchupResult,
    BenchmarkReport,
    CollectionMetrics,
    aggregate_benchmark_head_to_heads,
    benchmark_rollouts,
    collect_rollouts,
    default_benchmark_matchups,
    iter_rollout_records,
    policy_from_name,
    policy_from_spec,
    read_rollout_records,
    rollout_record_from_dict,
    rollout_record_to_dict,
    summarize_records,
)
from pokezero.env import StepResult, TerminalState
from pokezero.linear_policy import LinearPolicyModel, LinearSoftmaxPolicy, save_linear_model
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


class ResetFailingEnv:
    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        raise RuntimeError("boom")


class SeedRecordingEnv(OneTurnEnv):
    def __init__(self, reset_seeds: list[int]) -> None:
        super().__init__()
        self.reset_seeds = reset_seeds

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_seeds.append(seed)
        super().reset(seed=seed, format_id=format_id)


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
            streamed_records = list(iter_rollout_records(output_path))
        self.assertEqual(metrics.games, 2)
        self.assertEqual(metrics.p1_wins, 2)
        self.assertEqual(metrics.total_decision_rounds, 2)
        self.assertEqual([record.seed for record in records], [10, 11])
        self.assertEqual([record.seed for record in streamed_records], [10, 11])
        self.assertEqual([record.battle_id for record in records], ["rollout-10", "rollout-11"])

    def test_benchmark_rollouts_runs_default_matchups_without_writing_trajectories(self) -> None:
        report = benchmark_rollouts(
            games=2,
            env_factory=OneTurnEnv,
            rollout_config=RolloutConfig(max_decision_rounds=5),
            seed_start=20,
        )

        self.assertEqual(report.games_per_matchup, 2)
        self.assertEqual(report.total_games, 8)
        self.assertEqual(
            [result.label for result in report.matchups],
            [matchup.label for matchup in default_benchmark_matchups()],
        )
        for result in report.matchups:
            self.assertEqual(result.seed_start, 20)
            self.assertEqual(result.metrics.games, 2)
            self.assertEqual(result.metrics.p1_wins, 2)
            self.assertEqual(result.metrics.total_decision_rounds, 2)

    def test_benchmark_rollouts_reuses_seed_range_for_each_matchup(self) -> None:
        reset_seeds = []
        matchups = (
            BenchmarkMatchup("a", RandomLegalPolicy(), RandomLegalPolicy()),
            BenchmarkMatchup("b", RandomLegalPolicy(), RandomLegalPolicy()),
        )

        benchmark_rollouts(
            games=2,
            env_factory=lambda: SeedRecordingEnv(reset_seeds),
            rollout_config=RolloutConfig(max_decision_rounds=5),
            seed_start=7,
            matchups=matchups,
        )

        self.assertEqual(reset_seeds, [7, 8, 7, 8])

    def test_benchmark_head_to_head_aggregates_mirror_pair(self) -> None:
        rows = (
            BenchmarkMatchupResult(
                label="simple-legal vs random-legal",
                p1_policy_id="simple-legal",
                p2_policy_id="random-legal",
                seed_start=1,
                metrics=CollectionMetrics(
                    games=10,
                    elapsed_seconds=1.0,
                    total_decision_rounds=20,
                    total_simulator_turns=18,
                    p1_wins=6,
                    p2_wins=3,
                    ties=1,
                    capped_games=0,
                ),
            ),
            BenchmarkMatchupResult(
                label="random-legal vs simple-legal",
                p1_policy_id="random-legal",
                p2_policy_id="simple-legal",
                seed_start=1,
                metrics=CollectionMetrics(
                    games=10,
                    elapsed_seconds=1.0,
                    total_decision_rounds=20,
                    total_simulator_turns=18,
                    p1_wins=4,
                    p2_wins=5,
                    ties=0,
                    capped_games=1,
                ),
            ),
            BenchmarkMatchupResult(
                label="random-legal vs random-legal",
                p1_policy_id="random-legal",
                p2_policy_id="random-legal",
                seed_start=1,
                metrics=CollectionMetrics(
                    games=10,
                    elapsed_seconds=1.0,
                    total_decision_rounds=20,
                    total_simulator_turns=18,
                    p1_wins=5,
                    p2_wins=5,
                    ties=0,
                    capped_games=0,
                ),
            ),
        )

        head_to_heads = aggregate_benchmark_head_to_heads(rows)

        self.assertEqual(len(head_to_heads), 1)
        result = head_to_heads[0]
        self.assertEqual(result.label, "simple-legal vs random-legal")
        self.assertEqual(result.games, 20)
        self.assertEqual(result.first_policy_wins, 11)
        self.assertEqual(result.second_policy_wins, 7)
        self.assertEqual(result.ties, 1)
        self.assertEqual(result.capped_games, 1)
        self.assertAlmostEqual(result.first_policy_win_rate, 0.55)

    def test_collect_rollouts_non_append_preserves_existing_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"
            output_path.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "boom"):
                collect_rollouts(
                    output_path=output_path,
                    games=1,
                    env_factory=ResetFailingEnv,
                    policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                )

            self.assertEqual(output_path.read_text(encoding="utf-8"), "existing\n")

    def test_summarize_records_requires_explicit_elapsed_seconds(self) -> None:
        records = [collect_one_record_for_test()]

        metrics = summarize_records(records, elapsed_seconds=2.0)

        self.assertEqual(metrics.games, 1)
        self.assertEqual(metrics.elapsed_seconds, 2.0)
        self.assertEqual(metrics.games_per_second, 0.5)

    def test_policy_from_name_rejects_unknown_policy(self) -> None:
        self.assertEqual(policy_from_name("random-legal").policy_id, "random-legal")
        self.assertEqual(policy_from_name("simple-legal").policy_id, "simple-legal")
        with self.assertRaisesRegex(ValueError, "Unsupported policy"):
            policy_from_name("unknown")

    def test_policy_from_spec_loads_linear_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "linear.json"
            save_linear_model(
                checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="linear-test",
                ),
            )

            policy = policy_from_spec(f"linear:{checkpoint_path}")

        self.assertIsInstance(policy, LinearSoftmaxPolicy)
        self.assertEqual(policy.policy_id, "linear-test")

    def test_policy_from_spec_rejects_empty_linear_checkpoint_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint path"):
            policy_from_spec("linear:")

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

    def test_rollout_cli_collect_loads_linear_policy_spec(self) -> None:
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
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "linear.json"
            save_linear_model(
                checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="linear-cli-test",
                ),
            )
            with patch("pokezero.rollout_cli.collect_rollouts", return_value=fake_metrics) as collect:
                with patch("sys.stdout", new_callable=io.StringIO):
                    exit_code = rollout_cli_main(
                        [
                            "collect",
                            "--games",
                            "1",
                            "--out",
                            str(Path(temp_dir) / "rollouts.jsonl"),
                            "--p1-policy",
                            f"linear:{checkpoint_path}",
                            "--p2-policy",
                            "random-legal",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        kwargs = collect.call_args.kwargs
        self.assertEqual(kwargs["policies"]["p1"].policy_id, "linear-cli-test")
        self.assertEqual(kwargs["policies"]["p2"].policy_id, "random-legal")

    def test_rollout_cli_benchmark_wires_arguments_and_prints_report(self) -> None:
        fake_report = BenchmarkReport(
            format_id="gen3randombattle",
            max_decision_rounds=7,
            games_per_matchup=3,
            matchups=(
                BenchmarkMatchupResult(
                    label="random-legal vs random-legal",
                    p1_policy_id="random-legal",
                    p2_policy_id="random-legal",
                    seed_start=50,
                    metrics=CollectionMetrics(
                        games=3,
                        elapsed_seconds=2.0,
                        total_decision_rounds=12,
                        total_simulator_turns=9,
                        p1_wins=1,
                        p2_wins=2,
                        ties=0,
                        capped_games=0,
                    ),
                ),
            ),
        )
        with patch("pokezero.rollout_cli.benchmark_rollouts", return_value=fake_report) as benchmark:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = rollout_cli_main(
                    [
                        "benchmark",
                        "--games",
                        "3",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--seed-start",
                        "50",
                        "--max-decision-rounds",
                        "7",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = benchmark.call_args.kwargs
        self.assertEqual(kwargs["games"], 3)
        self.assertEqual(kwargs["seed_start"], 50)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 7)
        self.assertIn("total_games: 3", stdout.getvalue())
        self.assertIn("random-legal vs random-legal", stdout.getvalue())

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
