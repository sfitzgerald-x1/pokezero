import contextlib
import io
import json
import unittest

from pokezero.neural_cli import main as neural_cli_main
from pokezero.neural_policy import (
    NEURAL_INSTALL_MESSAGE,
    TorchUnavailableError,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    require_torch,
    torch_available,
    training_batch_to_torch,
)
from pokezero.showdown import ACTION_CANDIDATE_TOKEN_OFFSET, DEFAULT_REPLAY_OBSERVATION_SPEC


class NeuralPolicyScaffoldTest(unittest.TestCase):
    def test_transformer_policy_config_defaults_match_replay_observation_shape(self) -> None:
        config = TransformerPolicyConfig()

        self.assertEqual(config.window_size, 4)
        self.assertEqual(config.token_count, DEFAULT_REPLAY_OBSERVATION_SPEC.token_count)
        self.assertEqual(config.categorical_feature_count, DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count)
        self.assertEqual(config.numeric_feature_count, DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count)
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


if __name__ == "__main__":
    unittest.main()
