import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.env import TerminalState
from pokezero.neural_cli import main as neural_cli_main
from pokezero.neural_policy import (
    DEFAULT_CATEGORY_VOCAB_SIZE,
    DEFAULT_TOKEN_TYPE_VOCAB_SIZE,
    NEURAL_INSTALL_MESSAGE,
    TorchUnavailableError,
    TransformerSoftmaxPolicy,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    load_transformer_checkpoint,
    require_torch,
    save_transformer_checkpoint,
    torch_available,
    train_transformer_policy,
    training_batch_to_torch,
)
from pokezero.neural_selfplay import _require_promoted_opponent_pool as require_neural_promoted_opponent_pool
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.run_audit import RunAuditConfig, run_audit_config_payload
from pokezero.showdown import ACTION_CANDIDATE_TOKEN_OFFSET, CATEGORY_ID_BUCKETS, DEFAULT_REPLAY_OBSERVATION_SPEC
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


LEGAL_TWO_ACTION_MASK = (True, True, False, False, False, False, False, False, False)


def observation(value: int) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((value,) for _ in range(spec.token_count)),
        numeric_features=tuple((float(value),) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=LEGAL_TWO_ACTION_MASK,
    )


def rollout_record() -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id="neural-train", format_id="gen3randombattle", seed=10)
    for turn_index in range(4):
        action_index = turn_index % 2
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=turn_index,
                observation=observation(action_index + 1),
                legal_action_mask=LEGAL_TWO_ACTION_MASK,
                action_index=action_index,
                opponent_action_index=1 - action_index,
            )
        )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=4))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "fixture"},
        decision_round_count=4,
        elapsed_seconds=0.1,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


class NeuralPolicyScaffoldTest(unittest.TestCase):
    def test_transformer_policy_config_defaults_match_replay_observation_shape(self) -> None:
        config = TransformerPolicyConfig()

        self.assertEqual(config.window_size, 4)
        self.assertEqual(config.token_count, DEFAULT_REPLAY_OBSERVATION_SPEC.token_count)
        self.assertEqual(config.categorical_feature_count, DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count)
        self.assertEqual(config.numeric_feature_count, DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count)
        self.assertEqual(config.categorical_vocab_size, CATEGORY_ID_BUCKETS + 1)
        self.assertEqual(config.categorical_vocab_size, DEFAULT_CATEGORY_VOCAB_SIZE)
        self.assertEqual(config.token_type_vocab_size, DEFAULT_TOKEN_TYPE_VOCAB_SIZE)
        self.assertGreaterEqual(config.token_count, ACTION_CANDIDATE_TOKEN_OFFSET + 9)
        self.assertEqual(TransformerPolicyConfig.from_dict(config.to_dict()), config)

    def test_transformer_policy_config_validates_attention_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            TransformerPolicyConfig(embedding_dim=65, attention_heads=4)
        with self.assertRaisesRegex(ValueError, "token_count"):
            TransformerPolicyConfig(token_count=ACTION_CANDIDATE_TOKEN_OFFSET + 8)

    def test_transformer_training_config_validates_training_knobs(self) -> None:
        self.assertEqual(TransformerTrainingConfig().window_size, 4)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            TransformerTrainingConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "value_loss_weight"):
            TransformerTrainingConfig(value_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "opponent_action_loss_weight"):
            TransformerTrainingConfig(opponent_action_loss_weight=-0.1)

    def test_require_torch_fails_loudly_without_neural_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        with self.assertRaisesRegex(TorchUnavailableError, "pip install -e"):
            require_torch()

    def test_neural_promoted_opponent_pool_guard_does_not_require_torch(self) -> None:
        require_neural_promoted_opponent_pool(
            ("neural:a.pt", "neural:b.pt"),
            promotion_pool_registry_path=Path("promotions.json"),
            current_policy_spec="neural:b.pt",
            max_historical_opponents=2,
            required_size=1,
        )
        with self.assertRaisesRegex(ValueError, "promoted opponent pool has 1 selectable opponents.*required 2"):
            require_neural_promoted_opponent_pool(
                ("neural:a.pt", "neural:b.pt"),
                promotion_pool_registry_path=Path("promotions.json"),
                current_policy_spec="neural:b.pt",
                max_historical_opponents=2,
                required_size=2,
            )

    def test_tensor_conversion_fails_loudly_without_neural_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        with self.assertRaisesRegex(TorchUnavailableError, "pip install -e"):
            training_batch_to_torch(None)  # type: ignore[arg-type]

    def test_transformer_policy_construction_fails_loudly_without_neural_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        with self.assertRaisesRegex(TorchUnavailableError, "pip install -e"):
            TransformerSoftmaxPolicy(model=object(), result=None)  # type: ignore[arg-type]

    def test_neural_cli_describe_is_import_safe_without_torch(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = neural_cli_main(["describe", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["torch_available"], torch_available())
        self.assertEqual(payload["model_config"]["token_count"], DEFAULT_REPLAY_OBSERVATION_SPEC.token_count)

    def test_neural_cli_train_reports_missing_torch_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(["train", "--data", "missing.jsonl", "--out", "checkpoint.pt"])

        self.assertEqual(exit_code, 1)
        self.assertIn(NEURAL_INSTALL_MESSAGE, stderr.getvalue())

    def test_neural_cli_benchmark_reports_missing_torch_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(["benchmark", "--checkpoint", "checkpoint.pt", "--games", "1"])

        self.assertEqual(exit_code, 1)
        self.assertIn(NEURAL_INSTALL_MESSAGE, stderr.getvalue())

    def test_neural_cli_iterate_reports_missing_torch_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "1",
                    "--initial-policy",
                    "random-legal",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn(NEURAL_INSTALL_MESSAGE, stderr.getvalue())

    def test_neural_cli_benchmark_wires_fixed_baseline_matchups(self) -> None:
        class FakePolicy:
            policy_id = "neural-smoke"

        class FakeReport:
            def to_dict(self) -> dict:
                return {"ok": True}

        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        stdout = io.StringIO()
        fake_policy = FakePolicy()

        with (
            patch("pokezero.neural_cli._policy_from_checkpoint", return_value=fake_policy) as load,
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--games",
                    "2",
                    "--seed-start",
                    "44",
                    "--json",
                ]
            )

        matchups = captured["matchups"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(captured["games"], 2)
        self.assertEqual(captured["seed_start"], 44)
        self.assertEqual([matchup.label for matchup in matchups], [
            "neural-smoke vs random-legal",
            "random-legal vs neural-smoke",
            "neural-smoke vs simple-legal",
            "simple-legal vs neural-smoke",
        ])
        self.assertIs(matchups[0].p1_policy, fake_policy)
        self.assertIs(matchups[1].p2_policy, fake_policy)
        self.assertIs(matchups[2].p1_policy, fake_policy)
        self.assertIs(matchups[3].p2_policy, fake_policy)
        self.assertEqual(json.loads(stdout.getvalue()), {"ok": True})

    def test_neural_cli_iterate_wires_arguments(self) -> None:
        fake_epoch = type(
            "FakeEpoch",
            (),
            {"loss": 0.25, "policy_accuracy": 0.75},
        )()
        fake_iteration = type(
            "FakeIteration",
            (),
            {
                "iteration": 1,
                "metrics": type("FakeMetrics", (), {"games": 2})(),
                "checkpoint_path": Path("run/iteration-0001/transformer-policy.pt"),
                "training": type("FakeTraining", (), {"final_metrics": fake_epoch})(),
                "benchmark": None,
            },
        )()
        fake_result = type(
            "FakeResult",
            (),
            {
                "run_dir": Path("run"),
                "iterations": (fake_iteration,),
                "latest_checkpoint_path": Path("run/iteration-0001/transformer-policy.pt"),
                "to_dict": lambda self: {"ok": True},
            },
        )()
        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.run_neural_selfplay_iterations", return_value=fake_result) as run,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--workers",
                    "3",
                    "--resume",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--initial-policy",
                    "simple-legal",
                    "--opponent-policy",
                    "random-legal",
                    "--evaluation-games",
                    "4",
                    "--epochs",
                    "2",
                    "--batch-size",
                    "8",
                    "--policy-id",
                    "entity-cli",
                    "--promotion-registry",
                    "promotions.json",
                    "--require-promoted-opponent-pool-size",
                    "2",
                    "--auto-promote",
                    "--promotion-artifact-dir",
                    "promoted-checkpoints",
                    "--promotion-label-prefix",
                    "candidate",
                    "--promotion-notes",
                    "smoke notes",
                    "--allow-duplicate-promotion",
                    "--min-benchmark-win-rate",
                    "0.0",
                    "--min-benchmark-games",
                    "0",
                    "--audit-after-iteration",
                    "--audit-min-latest-benchmark-games",
                    "2",
                    "--audit-max-latest-average-decision-rounds",
                    "200",
                    "--audit-max-latest-benchmark-average-decision-rounds",
                    "210",
                    "--audit-max-latest-process-peak-rss-mb",
                    "2048",
                    "--audit-allow-missing-benchmark",
                    "--audit-allow-missing-benchmark-opponents",
                    "--json",
                ]
            )

        kwargs = run.call_args.kwargs
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"ok": True})
        self.assertEqual(kwargs["iterations"], 1)
        self.assertTrue(kwargs["resume"])
        self.assertEqual(kwargs["games_per_iteration"], 2)
        self.assertEqual(kwargs["initial_policy_spec"], "simple-legal")
        self.assertEqual(kwargs["fixed_opponent_policy_specs"], ("random-legal",))
        self.assertEqual(kwargs["evaluation_games"], 4)
        self.assertEqual(kwargs["worker_count"], 3)
        self.assertEqual(kwargs["training_config"].epochs, 2)
        self.assertEqual(kwargs["training_config"].batch_size, 8)
        self.assertEqual(kwargs["training_config"].capped_terminal_value, -0.25)
        self.assertEqual(kwargs["model_config"].policy_id, "entity-cli")
        self.assertEqual(kwargs["promotion_registry_path"], Path("promotions.json"))
        self.assertEqual(kwargs["required_promoted_opponent_pool_size"], 2)
        self.assertEqual(kwargs["auto_promotion_config"].registry_path, Path("promotions.json"))
        self.assertEqual(kwargs["auto_promotion_config"].artifact_dir, Path("promoted-checkpoints"))
        self.assertEqual(kwargs["auto_promotion_config"].label_prefix, "candidate")
        self.assertEqual(kwargs["auto_promotion_config"].notes, "smoke notes")
        self.assertTrue(kwargs["auto_promotion_config"].allow_duplicate)
        self.assertEqual(kwargs["auto_promotion_config"].gate_config.min_benchmark_win_rate, 0.0)
        self.assertEqual(kwargs["auto_promotion_config"].gate_config.min_benchmark_games, 0)
        self.assertEqual(kwargs["post_iteration_audit_config"].min_latest_benchmark_games, 2)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_latest_average_decision_rounds, 200.0)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_latest_benchmark_average_decision_rounds, 210.0)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_latest_process_peak_rss_mb, 2048.0)
        self.assertFalse(kwargs["post_iteration_audit_config"].require_benchmark)
        self.assertFalse(kwargs["post_iteration_audit_config"].require_benchmark_opponent_coverage)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_consecutive_promotion_failures, 3)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_benchmark_win_rate_drop, 0.15)

    def test_neural_cli_iterate_uses_named_post_iteration_audit_profile(self) -> None:
        fake_epoch = type(
            "FakeEpoch",
            (),
            {"loss": 0.25, "policy_accuracy": 0.75},
        )()
        fake_iteration = type(
            "FakeIteration",
            (),
            {
                "iteration": 1,
                "metrics": type("FakeMetrics", (), {"games": 2})(),
                "checkpoint_path": Path("run/iteration-0001/transformer-policy.pt"),
                "training": type("FakeTraining", (), {"final_metrics": fake_epoch})(),
                "benchmark": None,
            },
        )()
        fake_result = type(
            "FakeResult",
            (),
            {
                "run_dir": Path("run"),
                "iterations": (fake_iteration,),
                "latest_checkpoint_path": Path("run/iteration-0001/transformer-policy.pt"),
                "to_dict": lambda self: {"ok": True},
            },
        )()

        with (
            patch("pokezero.neural_cli.run_neural_selfplay_iterations", return_value=fake_result) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--initial-policy",
                    "random-legal",
                    "--evaluation-games",
                    "3",
                    "--audit-after-iteration",
                    "--audit-profile",
                    "long-run",
                    "--audit-min-latest-benchmark-games",
                    "7",
                    "--audit-allow-missing-benchmark",
                    "--audit-require-latest-promotion",
                ]
            )

        audit_config = run.call_args.kwargs["post_iteration_audit_config"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(audit_config.min_latest_benchmark_win_rate, 0.60)
        self.assertEqual(audit_config.min_latest_benchmark_games, 7)
        self.assertEqual(audit_config.max_latest_benchmark_capped_rate, 0.05)
        self.assertEqual(audit_config.max_benchmark_win_rate_drop, 0.03)
        self.assertFalse(audit_config.require_benchmark)
        self.assertTrue(audit_config.require_latest_promotion)
        self.assertTrue(audit_config.require_benchmark_opponent_coverage)

    def test_neural_cli_iterate_uses_post_iteration_audit_config_file(self) -> None:
        fake_epoch = type(
            "FakeEpoch",
            (),
            {"loss": 0.25, "policy_accuracy": 0.75},
        )()
        fake_iteration = type(
            "FakeIteration",
            (),
            {
                "iteration": 1,
                "metrics": type("FakeMetrics", (), {"games": 2})(),
                "checkpoint_path": Path("run/iteration-0001/transformer-policy.pt"),
                "training": type("FakeTraining", (), {"final_metrics": fake_epoch})(),
                "benchmark": None,
            },
        )()
        fake_result = type(
            "FakeResult",
            (),
            {
                "run_dir": Path("run"),
                "iterations": (fake_iteration,),
                "latest_checkpoint_path": Path("run/iteration-0001/transformer-policy.pt"),
                "to_dict": lambda self: {"ok": True},
            },
        )()
        audit_defaults = RunAuditConfig(
            min_latest_benchmark_win_rate=0.72,
            min_latest_benchmark_games=8,
            require_benchmark=True,
            require_benchmark_opponent_coverage=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "audit-config.json"
            config_path.write_text(json.dumps(run_audit_config_payload(audit_defaults), indent=2), encoding="utf-8")
            with (
                patch("pokezero.neural_cli.run_neural_selfplay_iterations", return_value=fake_result) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = neural_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--initial-policy",
                        "random-legal",
                        "--evaluation-games",
                        "2",
                        "--audit-after-iteration",
                        "--audit-config",
                        str(config_path),
                        "--audit-min-latest-benchmark-games",
                        "4",
                        "--audit-allow-missing-benchmark-opponents",
                    ]
                )

        audit_config = run.call_args.kwargs["post_iteration_audit_config"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(audit_config.min_latest_benchmark_win_rate, 0.72)
        self.assertEqual(audit_config.min_latest_benchmark_games, 4)
        self.assertTrue(audit_config.require_benchmark)
        self.assertFalse(audit_config.require_benchmark_opponent_coverage)

    def test_neural_cli_iterate_rejects_audit_requiring_missing_benchmark(self) -> None:
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--initial-policy",
                    "random-legal",
                    "--audit-after-iteration",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--audit-after-iteration requires --evaluation-games", stderr.getvalue())

    def test_neural_cli_iterate_rejects_audit_profile_with_too_few_evaluation_games(self) -> None:
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--initial-policy",
                    "random-legal",
                    "--evaluation-games",
                    "3",
                    "--audit-after-iteration",
                    "--audit-profile",
                    "long-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("requires enough --evaluation-games", stderr.getvalue())

    def test_neural_cli_help_lists_benchmark_command(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(stdout):
            neural_cli_main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("benchmark", stdout.getvalue())
        self.assertIn("iterate", stdout.getvalue())

    def test_torch_forward_train_save_load_and_policy_adapter_smoke(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            checkpoint_path = Path(temp_dir) / "transformer.pt"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())

            model, result = train_transformer_policy(
                data_path,
                model_config=TransformerPolicyConfig(
                    policy_id="neural-smoke",
                    window_size=2,
                    categorical_vocab_size=32,
                    token_type_vocab_size=8,
                    categorical_feature_count=1,
                    numeric_feature_count=1,
                    embedding_dim=16,
                    transformer_layers=1,
                    attention_heads=4,
                    feedforward_dim=32,
                    dropout=0.0,
                ),
                training_config=TransformerTrainingConfig(
                    batch_size=2,
                    epochs=1,
                    window_size=2,
                    max_batches=1,
                    device="cpu",
                ),
            )
            save_transformer_checkpoint(checkpoint_path, model, result=result)
            restored_model, restored_result = load_transformer_checkpoint(checkpoint_path, map_location="cpu")
            policy = TransformerSoftmaxPolicy(model=restored_model, result=restored_result, device="cpu")
            decision = policy.select_action(observation(1), rng=__import__("random").Random(1))
            _, continued_result = train_transformer_policy(
                data_path,
                model_config=TransformerPolicyConfig(
                    policy_id="neural-smoke-continued",
                    window_size=2,
                    categorical_vocab_size=32,
                    token_type_vocab_size=8,
                    categorical_feature_count=1,
                    numeric_feature_count=1,
                    embedding_dim=16,
                    transformer_layers=1,
                    attention_heads=4,
                    feedforward_dim=32,
                    dropout=0.0,
                ),
                training_config=TransformerTrainingConfig(
                    batch_size=2,
                    epochs=1,
                    window_size=2,
                    max_batches=1,
                    device="cpu",
                ),
                initial_model=restored_model,
            )

        self.assertEqual(result.final_metrics.examples, 2)
        self.assertEqual(continued_result.model_config.policy_id, "neural-smoke-continued")
        self.assertIn(decision.action_index, {0, 1})
        self.assertEqual(policy.policy_id, "neural-smoke")


if __name__ == "__main__":
    unittest.main()
