"""HIGH-1 latch tests: checkpoint-stamped feature masks must be read back into every
env-construction-from-checkpoint path (the mask-axis twin of the #492 belief mismatch)."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pokezero.local_showdown import LocalShowdownConfig, env_config_with_checkpoint_masks
from pokezero.observation import (
    DEFAULT_OBSERVATION_FEATURE_MASKS,
    TRANSITION_TOKEN_COUNT,
    ObservationFeatureMasks,
)

K32_MASKS = ObservationFeatureMasks(transition_token_budget=32)
STATS_OFF_MASKS = ObservationFeatureMasks(stats_block=False)


class EnvConfigMaskResolutionTest(unittest.TestCase):
    def test_no_transformer_checkpoints_leaves_config_unchanged(self) -> None:
        config = LocalShowdownConfig()
        self.assertIs(env_config_with_checkpoint_masks(config, (), context="t"), config)

    def test_default_env_adopts_the_checkpoint_masks(self) -> None:
        config = LocalShowdownConfig()
        resolved = env_config_with_checkpoint_masks(config, K32_MASKS, context="t")
        self.assertEqual(resolved.feature_masks, K32_MASKS)

    def test_matching_masks_are_a_no_op(self) -> None:
        config = LocalShowdownConfig(feature_masks=K32_MASKS)
        resolved = env_config_with_checkpoint_masks(config, (K32_MASKS, K32_MASKS), context="t")
        self.assertIs(resolved, config)

    def test_conflicting_checkpoints_hard_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting observation feature masks"):
            env_config_with_checkpoint_masks(
                LocalShowdownConfig(), (K32_MASKS, STATS_OFF_MASKS), context="t"
            )

    def test_explicit_env_override_conflicting_with_checkpoint_hard_fails(self) -> None:
        config = LocalShowdownConfig(feature_masks=STATS_OFF_MASKS)
        with self.assertRaisesRegex(ValueError, "conflict with the loaded checkpoint"):
            env_config_with_checkpoint_masks(config, K32_MASKS, context="t")

    def test_full_default_checkpoint_keeps_default_env(self) -> None:
        config = LocalShowdownConfig()
        resolved = env_config_with_checkpoint_masks(
            config, DEFAULT_OBSERVATION_FEATURE_MASKS, context="t"
        )
        self.assertEqual(resolved.feature_masks, DEFAULT_OBSERVATION_FEATURE_MASKS)


def _torch_available() -> bool:
    from pokezero.neural_policy import torch_available

    return torch_available()


def _save_k32_checkpoint(path: Path):
    """A real saved checkpoint whose model config carries the K=32 ablation budget."""
    from pokezero.neural_policy import (
        EntityTokenTransformerPolicy,
        TransformerPolicyConfig,
        TransformerTrainingConfig,
        TransformerTrainingResult,
        save_transformer_checkpoint,
    )

    config = TransformerPolicyConfig.compact_category(
        policy_id="k32-arm",
        category_vocab=tuple(f"token-{index}" for index in range(8)),
        category_oov_buckets=2,
        window_size=1,
        embedding_dim=8,
        transformer_layers=0,
        attention_heads=1,
        feedforward_dim=8,
        dropout=0.0,
        transition_token_budget=32,
    )
    model = EntityTokenTransformerPolicy(config)
    result = TransformerTrainingResult(
        model_config=config,
        training_config=TransformerTrainingConfig(window_size=1),
        epochs=(),
    )
    save_transformer_checkpoint(path, model, result=result)
    return config


class MaskDerivationTest(unittest.TestCase):
    def test_feature_masks_from_model_config_round_trips(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import TransformerPolicyConfig, feature_masks_from_model_config

        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",),
            category_oov_buckets=2,
            stats_block_enabled=False,
            exact_state_enabled=True,
            transition_token_budget=32,
        )
        masks = feature_masks_from_model_config(config)
        self.assertEqual(
            masks,
            ObservationFeatureMasks(
                stats_block=False, exact_state=True, transition_token_budget=32
            ),
        )
        default_config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2
        )
        self.assertEqual(
            feature_masks_from_model_config(default_config), DEFAULT_OBSERVATION_FEATURE_MASKS
        )
        self.assertEqual(default_config.transition_token_budget, TRANSITION_TOKEN_COUNT)

    def test_transformer_policy_sweep_finds_model_configs(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import (
            TransformerPolicyConfig,
            TransformerTrainingConfig,
            TransformerTrainingResult,
            transformer_model_configs_from_policies,
        )
        from pokezero.policy import RandomLegalPolicy

        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",), category_oov_buckets=2, transition_token_budget=32
        )

        class _FakeNeuralPolicy:
            result = TransformerTrainingResult(
                model_config=config,
                training_config=TransformerTrainingConfig(window_size=1),
                epochs=(),
            )

        configs = transformer_model_configs_from_policies(
            [RandomLegalPolicy(), _FakeNeuralPolicy(), object()]
        )
        self.assertEqual(configs, (config,))


class K32HarnessPathTest(unittest.TestCase):
    """Each harness path must build a K=32 env for a K=32 checkpoint."""

    def test_neural_cli_benchmark_builds_k32_env(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_cli import main as neural_cli_main

        captured: dict[str, object] = {}

        def fake_benchmark_rollouts(*, games, env_factory, rollout_config, seed_start, matchups):
            captured["env"] = env_factory()

            class _Report:
                def to_dict(self):
                    return {}

            return _Report()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            with (
                patch("pokezero.neural_cli.benchmark_rollouts", fake_benchmark_rollouts),
                patch("pokezero.neural_cli.print_benchmark_report"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = neural_cli_main(
                    [
                        "benchmark",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--games",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
        self.assertEqual(exit_code, 0)
        env = captured["env"]
        self.assertEqual(env.config.feature_masks, K32_MASKS)

    def test_policy_spec_resolver_builds_k32_env_config(self) -> None:
        # The shared path used by rollout_cli collect/benchmark/replay and the bootstrap
        # teacher harnesses: neural: specs contribute their stamped masks.
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.collection import env_config_with_policy_spec_masks

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            resolved = env_config_with_policy_spec_masks(
                LocalShowdownConfig(),
                (f"neural:{checkpoint_path}", "random-legal", None),
                context="spec harness",
            )
        self.assertEqual(resolved.feature_masks, K32_MASKS)

    def test_neural_cli_spec_mask_helper_covers_iterate_and_root_puct_paths(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_cli import _env_config_with_spec_masks
        from pokezero.neural_policy import load_transformer_model_config

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            loaded_config = load_transformer_model_config(checkpoint_path)
            # iterate / root-puct shape: a directly loaded model config plus policy specs.
            resolved = _env_config_with_spec_masks(
                LocalShowdownConfig(),
                (f"neural:{checkpoint_path}", "random-legal"),
                extra_model_configs=(loaded_config,),
                context="iterate",
            )
            self.assertEqual(resolved.feature_masks, K32_MASKS)
            # A conflicting full-default checkpoint alongside the K=32 arm must hard-fail.
            from pokezero.neural_policy import TransformerPolicyConfig

            default_config = TransformerPolicyConfig.compact_category(
                category_vocab=("species:a",), category_oov_buckets=2
            )
            with self.assertRaisesRegex(ValueError, "conflicting observation feature masks"):
                _env_config_with_spec_masks(
                    LocalShowdownConfig(),
                    (f"neural:{checkpoint_path}",),
                    extra_model_configs=(default_config,),
                    context="iterate",
                )

    def test_build_agent_carries_k32_masks_for_online_and_factor_paths(self) -> None:
        if not _torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.category_vocab import build_category_vocabulary
        from pokezero.online_client import build_agent

        fake_vocab = build_category_vocabulary(["species:a"], oov_buckets=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "k32.pt"
            _save_k32_checkpoint(checkpoint_path)
            with (
                patch("pokezero.randbat_vocab.gen3_category_vocabulary", return_value=fake_vocab),
                patch("pokezero.dex.load_showdown_dex_cached", return_value=object()),
            ):
                agent = build_agent(checkpoint_path, temp_dir, our_name="bot")
        self.assertEqual(agent.feature_masks, K32_MASKS)


if __name__ == "__main__":
    unittest.main()
