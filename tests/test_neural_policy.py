import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

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
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
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

        self.assertEqual(result.final_metrics.examples, 2)
        self.assertIn(decision.action_index, {0, 1})
        self.assertEqual(policy.policy_id, "neural-smoke")


if __name__ == "__main__":
    unittest.main()
