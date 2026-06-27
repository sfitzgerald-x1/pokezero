import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.env import TerminalState
from pokezero.neural_cli import main as neural_cli_main
from pokezero.neural_policy import (
    DEFAULT_TOKEN_TYPE_VOCAB_SIZE,
    NEURAL_INSTALL_MESSAGE,
    EntityTokenTransformerPolicy,
    TorchUnavailableError,
    TransformerSoftmaxPolicy,
    TransformerEpochMetrics,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
    evaluate_transformer_opponent_action_priors,
    load_transformer_checkpoint,
    require_torch,
    resolve_torch_device,
    save_transformer_checkpoint,
    torch_available,
    train_transformer_policy,
    training_batch_to_torch,
    _greedy_action_index,
)
from pokezero.neural_selfplay import _require_promoted_opponent_pool as require_neural_promoted_opponent_pool
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.policy import PolicyContext
from pokezero.run_audit import RunAuditConfig, run_audit_config_payload
from pokezero.showdown import ACTION_CANDIDATE_TOKEN_OFFSET, DEFAULT_REPLAY_OBSERVATION_SPEC
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


LEGAL_TWO_ACTION_MASK = (True, True, False, False, False, False, False, False, False)
LEGAL_ACTION_ONE_MASK = (False, True, False, False, False, False, False, False, False)
LEGAL_ACTION_ONE_TWO_MASK = (False, True, True, False, False, False, False, False, False)


def observation(
    value: int,
    *,
    legal_action_mask: tuple[bool, ...] = LEGAL_TWO_ACTION_MASK,
) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((value,) for _ in range(spec.token_count)),
        numeric_features=tuple((float(value),) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=legal_action_mask,
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
    def setUp(self) -> None:
        # Self-play (iterate) builds the string->row CategoryVocabulary from --showdown-root;
        # stub it so CLI tests stay fast without a real Showdown checkout.
        from pokezero.category_vocab import build_category_vocabulary

        fake_vocab = build_category_vocabulary(["species:a", "species:b", "move:c"], oov_buckets=16)
        vocab_patch = patch("pokezero.randbat_vocab.gen3_category_vocabulary", return_value=fake_vocab)
        vocab_patch.start()
        self.addCleanup(vocab_patch.stop)

    def test_transformer_policy_config_defaults_match_replay_observation_shape(self) -> None:
        config = TransformerPolicyConfig.compact_category(category_vocab=(1, 2, 3), category_oov_buckets=4)

        self.assertEqual(config.window_size, 4)
        self.assertEqual(config.token_count, DEFAULT_REPLAY_OBSERVATION_SPEC.token_count)
        self.assertEqual(config.categorical_feature_count, DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count)
        self.assertEqual(config.numeric_feature_count, DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count)
        # categorical_vocab_size is derived: 1 padding + 3 vocab + 4 oov.
        self.assertEqual(config.categorical_vocab_size, 8)
        self.assertEqual(config.token_type_vocab_size, DEFAULT_TOKEN_TYPE_VOCAB_SIZE)
        self.assertEqual(config.value_activation, "tanh")
        self.assertGreaterEqual(config.token_count, ACTION_CANDIDATE_TOKEN_OFFSET + 9)
        self.assertEqual(TransformerPolicyConfig.from_dict(config.to_dict()), config)

    def test_transformer_policy_config_loads_legacy_value_activation_as_linear(self) -> None:
        config = TransformerPolicyConfig.compact_category(category_vocab=(1, 2, 3), category_oov_buckets=4)
        payload = config.to_dict()
        payload.pop("value_activation")

        restored = TransformerPolicyConfig.from_dict(payload)

        self.assertEqual(restored.value_activation, "linear")

    def test_transformer_policy_config_requires_category_vocab(self) -> None:
        with self.assertRaisesRegex(ValueError, "category_vocab is required"):
            TransformerPolicyConfig()

    def test_validate_initial_model_config_detects_warm_start_vocab_mismatch(self) -> None:
        from pokezero.neural_policy import _validate_initial_model_config

        base = TransformerPolicyConfig.compact_category(category_vocab=(1, 2, 3), category_oov_buckets=4)
        other = TransformerPolicyConfig.compact_category(category_vocab=(1, 2, 3, 4), category_oov_buckets=4)
        # Same config except policy_id is allowed (warm-start of the same embedding).
        _validate_initial_model_config(SimpleNamespace(config=replace(base, policy_id="warm")), base)
        # A different category vocabulary must be rejected (the retired-format resume guard).
        with self.assertRaises(ValueError):
            _validate_initial_model_config(SimpleNamespace(config=other), base)
        # Models without a config (e.g. a non-neural collector) are skipped, not rejected.
        _validate_initial_model_config(SimpleNamespace(config=None), base)
        _validate_initial_model_config(object(), base)

    def test_transformer_policy_config_validates_attention_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            TransformerPolicyConfig.compact_category(category_vocab=(1,), category_oov_buckets=1, embedding_dim=65, attention_heads=4)
        with self.assertRaisesRegex(ValueError, "token_count"):
            TransformerPolicyConfig.compact_category(category_vocab=(1,), category_oov_buckets=1, token_count=ACTION_CANDIDATE_TOKEN_OFFSET + 8)
        with self.assertRaisesRegex(ValueError, "value_activation"):
            TransformerPolicyConfig.compact_category(category_vocab=(1,), category_oov_buckets=1, value_activation="sigmoid")

    def test_evaluate_transformer_observation_value_uses_configured_history_window(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
        config = TransformerPolicyConfig.compact_category(
            policy_id="fixture",
            category_vocab=("fixture",),
            category_oov_buckets=1,
            window_size=2,
            categorical_feature_count=1,
            numeric_feature_count=1,
            token_count=spec.token_count,
            embedding_dim=4,
            transformer_layers=1,
            attention_heads=1,
            feedforward_dim=8,
        )

        class FakeValueModel:
            def __init__(self) -> None:
                self.eval_called = False
                self.shapes: dict[str, tuple[int, ...]] = {}

            def eval(self) -> None:
                self.eval_called = True

            def __call__(self, **kwargs):
                self.shapes = {name: tuple(value.shape) for name, value in kwargs.items()}
                return SimpleNamespace(value=torch.tensor([0.42]))

        model = FakeValueModel()

        value = evaluate_transformer_observation_value(
            model=model,
            result=SimpleNamespace(model_config=config),
            observations=(observation(1), observation(2), observation(3)),
            device="cpu",
        )

        self.assertAlmostEqual(value, 0.42, places=5)
        self.assertTrue(model.eval_called)
        self.assertEqual(model.shapes["categorical_ids"], (1, 2, spec.token_count, 1))
        self.assertEqual(model.shapes["numeric_features"], (1, 2, spec.token_count, 1))
        self.assertEqual(model.shapes["history_mask"], (1, 2))

    def test_transformer_value_output_is_bounded_to_terminal_return_range(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",),
            category_oov_buckets=1,
            embedding_dim=8,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=16,
            dropout=0.0,
        )
        model = EntityTokenTransformerPolicy(config)
        shape = (1, config.window_size, config.token_count)
        inputs = {
            "categorical_ids": torch.zeros((*shape, config.categorical_feature_count), dtype=torch.long),
            "numeric_features": torch.zeros((*shape, config.numeric_feature_count), dtype=torch.float32),
            "token_type_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.bool),
            "history_mask": torch.ones((1, config.window_size), dtype=torch.bool),
        }

        with torch.no_grad():
            model.value_head.weight.zero_()
            model.value_head.bias.fill_(5.0)
        output = model(**inputs)
        bounded_value = float(output.value[0].detach())
        self.assertLess(bounded_value, 1.0)
        self.assertGreater(bounded_value, 0.99)

        with torch.no_grad():
            model.value_head.bias.fill_(-5.0)
        output = model(**inputs)
        bounded_value = float(output.value[0].detach())
        self.assertGreater(bounded_value, -1.0)
        self.assertLess(bounded_value, -0.99)

    def test_transformer_linear_value_activation_preserves_unbounded_outputs(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        config = replace(
            TransformerPolicyConfig.compact_category(
                category_vocab=("species:a",),
                category_oov_buckets=1,
                embedding_dim=8,
                transformer_layers=1,
                attention_heads=2,
                feedforward_dim=16,
                dropout=0.0,
            ),
            value_activation="linear",
        )
        model = EntityTokenTransformerPolicy(config)
        shape = (1, config.window_size, config.token_count)
        inputs = {
            "categorical_ids": torch.zeros((*shape, config.categorical_feature_count), dtype=torch.long),
            "numeric_features": torch.zeros((*shape, config.numeric_feature_count), dtype=torch.float32),
            "token_type_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.bool),
            "history_mask": torch.ones((1, config.window_size), dtype=torch.bool),
        }

        with torch.no_grad():
            model.value_head.weight.zero_()
            model.value_head.bias.fill_(5.0)

        output = model(**inputs)

        self.assertGreater(float(output.value[0].detach()), 4.99)

    def test_evaluate_transformer_action_priors_masks_illegal_actions(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
        config = TransformerPolicyConfig.compact_category(
            policy_id="fixture",
            category_vocab=("fixture",),
            category_oov_buckets=1,
            window_size=2,
            categorical_feature_count=1,
            numeric_feature_count=1,
            token_count=spec.token_count,
            embedding_dim=4,
            transformer_layers=1,
            attention_heads=1,
            feedforward_dim=8,
        )

        class FakePriorModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                logits = torch.zeros(1, 9)
                logits[0, 0] = -2.0
                logits[0, 1] = 2.0
                logits[0, 2] = 20.0
                return SimpleNamespace(policy_logits=logits, value=torch.tensor([0.0]))

        priors = evaluate_transformer_action_priors(
            model=FakePriorModel(),
            result=SimpleNamespace(model_config=config),
            observations=(observation(1), observation(2), observation(3)),
            device="cpu",
        )

        self.assertEqual(len(priors), 9)
        self.assertAlmostEqual(sum(priors), 1.0, places=5)
        self.assertGreater(priors[1], priors[0])
        self.assertEqual(priors[2], 0.0)

    def test_transformer_training_config_validates_training_knobs(self) -> None:
        self.assertEqual(TransformerTrainingConfig().window_size, 4)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            TransformerTrainingConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "value_loss_weight"):
            TransformerTrainingConfig(value_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "opponent_action_loss_weight"):
            TransformerTrainingConfig(opponent_action_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "switch_action_loss_weight"):
            TransformerTrainingConfig(switch_action_loss_weight=0.0)
        with self.assertRaisesRegex(ValueError, "action_family_loss_weight"):
            TransformerTrainingConfig(action_family_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "switch_target_loss_weight"):
            TransformerTrainingConfig(switch_target_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "objective"):
            TransformerTrainingConfig(objective="bogus")
        with self.assertRaisesRegex(ValueError, "clip_epsilon"):
            TransformerTrainingConfig(objective="ppo", clip_epsilon=0.0)
        with self.assertRaisesRegex(ValueError, "freeze_non_value_parameters"):
            TransformerTrainingConfig(freeze_non_value_parameters=True)
        # round-trips through to_dict/from_dict-equivalent (asdict) with RL knobs.
        self.assertEqual(TransformerTrainingConfig(objective="ppo").objective, "ppo")
        self.assertEqual(TransformerTrainingConfig(objective="reward-weighted").objective, "reward-weighted")
        self.assertEqual(TransformerTrainingConfig(objective="value-only", freeze_non_value_parameters=True).objective, "value-only")

    def test_ppo_objective_uses_value_baselined_clipped_surrogate(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        # Uniform logits over all-legal actions -> current prob == behavior prob (ratio 1);
        # returns 1 with value 0 -> advantage +1, so the clipped surrogate pushes chosen-action
        # prob up and the policy loss is negative.
        output = TransformerPolicyOutput(
            policy_logits=torch.zeros(3, 9),
            value=torch.zeros(3),
            opponent_action_logits=torch.zeros(3, 9),
        )
        tensors = {
            "legal_action_mask": torch.ones(3, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0, 1, 2], dtype=torch.long),
            "returns": torch.ones(3),
            "action_probabilities": torch.full((3,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(3, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(3, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(3, dtype=torch.long),
        }
        config = TransformerTrainingConfig(
            objective="ppo", normalize_advantage=False, entropy_coef=0.0, opponent_action_loss_weight=0.0
        )
        loss, metrics = _transformer_loss(output, tensors, config)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(metrics["policy_loss"], 0.0)  # positive advantage -> negative policy loss
        self.assertAlmostEqual(metrics["value_loss"], 1.0, places=5)  # MSE(0, 1)
        # Behavior-cloning objective on the same tensors yields a positive CE policy loss.
        bc_loss, bc_metrics = _transformer_loss(output, tensors, TransformerTrainingConfig())
        self.assertGreater(bc_metrics["policy_loss"], 0.0)

    def test_reward_weighted_objective_ignores_non_positive_returns(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        logits = torch.zeros(2, 9)
        logits[1, 0] = 20.0  # Badly wrong for target action 1, but this row has negative return.
        output = TransformerPolicyOutput(
            policy_logits=logits,
            value=torch.zeros(2),
            opponent_action_logits=torch.zeros(2, 9),
        )
        tensors = {
            "legal_action_mask": torch.ones(2, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0, 1], dtype=torch.long),
            "returns": torch.tensor([1.0, -1.0]),
            "action_probabilities": torch.full((2,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(2, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(2, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(2, dtype=torch.long),
        }
        _, weighted_metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(objective="reward-weighted", opponent_action_loss_weight=0.0),
        )
        _, bc_metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(objective="behavior-cloning", opponent_action_loss_weight=0.0),
        )

        self.assertAlmostEqual(weighted_metrics["policy_loss"], torch.log(torch.tensor(9.0)).item(), places=5)
        self.assertGreater(bc_metrics["policy_loss"], weighted_metrics["policy_loss"])

    def test_value_only_objective_skips_policy_and_auxiliary_losses(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        output = TransformerPolicyOutput(
            policy_logits=torch.full((2, 9), 10.0),
            value=torch.zeros(2),
            opponent_action_logits=torch.full((2, 9), 10.0),
        )
        tensors = {
            "legal_action_mask": torch.ones(2, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0, 4], dtype=torch.long),
            "returns": torch.ones(2),
            "action_probabilities": torch.full((2,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(2, dtype=torch.bool),
            "opponent_action_mask": torch.ones(2, dtype=torch.bool),
            "opponent_action_indices": torch.tensor([1, 2], dtype=torch.long),
        }

        loss, metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(
                objective="value-only",
                opponent_action_loss_weight=10.0,
                action_family_loss_weight=10.0,
                switch_target_loss_weight=10.0,
            ),
        )

        self.assertAlmostEqual(float(loss.detach().item()), 1.0, places=5)
        self.assertEqual(metrics["policy_loss"], 0.0)
        self.assertEqual(metrics["value_loss"], 1.0)
        self.assertEqual(metrics["opponent_examples"], 0)
        self.assertEqual(metrics["action_family_examples"], 0)
        self.assertEqual(metrics["switch_target_examples"], 0)

    def test_behavior_cloning_can_upweight_switch_action_labels(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        logits = torch.zeros(2, 9)
        logits[1, 0] = 20.0  # Very wrong for switch target action 4.
        output = TransformerPolicyOutput(
            policy_logits=logits,
            value=torch.zeros(2),
            opponent_action_logits=torch.zeros(2, 9),
        )
        tensors = {
            "legal_action_mask": torch.ones(2, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0, 4], dtype=torch.long),
            "returns": torch.ones(2),
            "action_probabilities": torch.full((2,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(2, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(2, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(2, dtype=torch.long),
        }

        _, unweighted = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(objective="behavior-cloning", opponent_action_loss_weight=0.0),
        )
        _, weighted = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(
                objective="behavior-cloning",
                opponent_action_loss_weight=0.0,
                switch_action_loss_weight=4.0,
            ),
        )

        self.assertGreater(weighted["policy_loss"], unweighted["policy_loss"])

    def test_action_family_loss_trains_move_vs_switch_decisions(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        logits = torch.zeros(2, 9)
        logits[0, 0] = 5.0  # Correct move-family example.
        logits[1, 0] = 5.0  # Wrong family for switch target action 4.
        output = TransformerPolicyOutput(
            policy_logits=logits,
            value=torch.zeros(2),
            opponent_action_logits=torch.zeros(2, 9),
        )
        tensors = {
            "legal_action_mask": torch.ones(2, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0, 4], dtype=torch.long),
            "returns": torch.ones(2),
            "action_probabilities": torch.full((2,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(2, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(2, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(2, dtype=torch.long),
        }

        base_loss, base_metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(objective="behavior-cloning", opponent_action_loss_weight=0.0),
        )
        family_loss, family_metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(
                objective="behavior-cloning",
                opponent_action_loss_weight=0.0,
                action_family_loss_weight=0.5,
            ),
        )

        self.assertEqual(base_metrics["action_family_examples"], 0)
        self.assertEqual(family_metrics["action_family_examples"], 2)
        self.assertEqual(family_metrics["action_family_correct"], 1)
        self.assertGreater(family_metrics["action_family_loss"], 0.0)
        self.assertGreater(float(family_loss.detach().item()), float(base_loss.detach().item()))

    def test_switch_target_loss_trains_conditional_switch_selection(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        logits = torch.zeros(3, 9)
        logits[0, 4] = 5.0  # Correct switch target.
        logits[1, 4] = 5.0  # Wrong switch target; teacher chose action 5.
        logits[2, 0] = 5.0  # Move examples do not contribute to switch-target aux loss.
        output = TransformerPolicyOutput(
            policy_logits=logits,
            value=torch.zeros(3),
            opponent_action_logits=torch.zeros(3, 9),
        )
        tensors = {
            "legal_action_mask": torch.ones(3, 9, dtype=torch.bool),
            "action_indices": torch.tensor([4, 5, 0], dtype=torch.long),
            "returns": torch.ones(3),
            "action_probabilities": torch.full((3,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(3, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(3, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(3, dtype=torch.long),
        }

        base_loss, base_metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(objective="behavior-cloning", opponent_action_loss_weight=0.0),
        )
        switch_loss, switch_metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(
                objective="behavior-cloning",
                opponent_action_loss_weight=0.0,
                switch_target_loss_weight=0.5,
            ),
        )

        self.assertEqual(base_metrics["switch_target_examples"], 0)
        self.assertEqual(switch_metrics["switch_target_examples"], 2)
        self.assertEqual(switch_metrics["switch_target_correct"], 1)
        self.assertGreater(switch_metrics["switch_target_loss"], 0.0)
        self.assertGreater(float(switch_loss.detach().item()), float(base_loss.detach().item()))

    def test_reward_weighted_objective_has_zero_policy_loss_without_positive_returns(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        output = TransformerPolicyOutput(
            policy_logits=torch.zeros(2, 9),
            value=torch.zeros(2),
            opponent_action_logits=torch.zeros(2, 9),
        )
        tensors = {
            "legal_action_mask": torch.ones(2, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0, 1], dtype=torch.long),
            "returns": torch.tensor([0.0, -1.0]),
            "action_probabilities": torch.full((2,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(2, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(2, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(2, dtype=torch.long),
        }
        loss, metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(objective="reward-weighted", opponent_action_loss_weight=0.0),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["policy_loss"], 0.0)

    def test_family_gated_greedy_selects_family_before_action(self) -> None:
        probabilities = (0.40, 0.01, 0.01, 0.01, 0.15, 0.14, 0.13, 0.12, 0.03)
        legal = tuple(range(9))

        self.assertEqual(_greedy_action_index(probabilities=probabilities, legal=legal, family_gated=False), 0)
        self.assertEqual(_greedy_action_index(probabilities=probabilities, legal=legal, family_gated=True), 4)

    def test_ppo_clips_large_positive_advantage_ratio(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        # Current policy strongly favors action 0 (prob ~1) while the behavior prob was 0.01,
        # so ratio ~100. With clip 0.2 and advantage +1, the surrogate must be clipped to
        # (1+0.2)*1 = 1.2 -> policy_loss ~ -1.2, NOT the unclipped ~-100.
        logits = torch.zeros(1, 9)
        logits[0, 0] = 20.0
        output = TransformerPolicyOutput(policy_logits=logits, value=torch.zeros(1), opponent_action_logits=torch.zeros(1, 9))
        tensors = {
            "legal_action_mask": torch.ones(1, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0], dtype=torch.long),
            "returns": torch.ones(1),
            "action_probabilities": torch.full((1,), 0.01),
            "action_probability_mask": torch.ones(1, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(1, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(1, dtype=torch.long),
        }
        config = TransformerTrainingConfig(objective="ppo", normalize_advantage=False, opponent_action_loss_weight=0.0, clip_epsilon=0.2)
        _, metrics = _transformer_loss(output, tensors, config)
        self.assertAlmostEqual(metrics["policy_loss"], -1.2, places=2)

    def test_ppo_masks_examples_without_positive_behavior_prob(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        output = TransformerPolicyOutput(policy_logits=torch.zeros(2, 9), value=torch.zeros(2), opponent_action_logits=torch.zeros(2, 9))
        config = TransformerTrainingConfig(objective="ppo", normalize_advantage=False, opponent_action_loss_weight=0.0)
        base = {
            "legal_action_mask": torch.ones(2, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0, 1], dtype=torch.long),
            "returns": torch.ones(2),
            "opponent_action_mask": torch.zeros(2, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(2, dtype=torch.long),
        }
        # All examples masked out (no recorded prob) -> zero policy loss, finite total loss.
        all_masked = {**base, "action_probabilities": torch.full((2,), 1.0 / 9.0), "action_probability_mask": torch.zeros(2, dtype=torch.bool)}
        loss, metrics = _transformer_loss(output, all_masked, config)
        self.assertEqual(metrics["policy_loss"], 0.0)
        self.assertTrue(torch.isfinite(loss))
        # A zero behavior probability is excluded even if its mask flag is set.
        zero_prob = {**base, "action_probabilities": torch.tensor([0.0, 1.0 / 9.0]), "action_probability_mask": torch.ones(2, dtype=torch.bool)}
        loss2, _ = _transformer_loss(output, zero_prob, config)
        self.assertTrue(torch.isfinite(loss2))

    def test_behavior_probability_mixes_epsilon_for_sampling(self) -> None:
        from pokezero.neural_policy import _behavior_probability

        # Sampling branch: (1 - eps) * pi(a) + eps / |legal|.
        self.assertAlmostEqual(
            _behavior_probability(action_index=0, probabilities=[0.5, 0.5], legal=[0, 1], deterministic=False, greedy_action=0, exploration_epsilon=0.2),
            0.8 * 0.5 + 0.2 / 2,
        )
        # epsilon == 0 reduces to pi(a).
        self.assertAlmostEqual(
            _behavior_probability(action_index=0, probabilities=[0.7, 0.3], legal=[0, 1], deterministic=False, greedy_action=0, exploration_epsilon=0.0),
            0.7,
        )

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

    def test_resolve_torch_device_matches_training_default(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        self.assertEqual(resolve_torch_device("cpu"), "cpu")
        with patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_torch_device(None), "cuda")
            self.assertEqual(resolve_torch_device(""), "cuda")
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(resolve_torch_device(None), "cpu")

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

    def test_neural_cli_value_calibration_reports_missing_torch_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(
                ["value-calibration", "--checkpoint", "checkpoint.pt", "--data", "rollouts.jsonl"]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn(NEURAL_INSTALL_MESSAGE, stderr.getvalue())

    def test_neural_cli_root_puct_benchmark_reports_missing_torch_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(["root-puct-benchmark", "--checkpoint", "checkpoint.pt", "--games", "1"])

        self.assertEqual(exit_code, 1)
        self.assertIn(NEURAL_INSTALL_MESSAGE, stderr.getvalue())

    def test_neural_cli_root_puct_counterfactual_reports_missing_torch_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(["root-puct-counterfactual", "--checkpoint", "checkpoint.pt", "--games", "1"])

        self.assertEqual(exit_code, 1)
        self.assertIn(NEURAL_INSTALL_MESSAGE, stderr.getvalue())

    def test_neural_cli_root_puct_play_benchmark_reports_missing_torch_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(["root-puct-play-benchmark", "--checkpoint", "checkpoint.pt", "--games", "1"])

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

    def test_neural_cli_root_puct_play_benchmark_wires_raw_and_search_matchups(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 4}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1))
        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            matchups = tuple(kwargs["matchups"])
            search_policy = matchups[2].p1_policy
            self.assertEqual(search_policy.value_fn((observation(1),)), 0.25)
            self.assertEqual(search_policy.prior_fn((observation(1),)), (1.0,) + (0.0,) * 8)
            context = PolicyContext(
                player_id="p1",
                decision_round_index=0,
                battle_id="search-play",
                format_id="gen3randombattle",
                seed=7,
                observation=observation(1),
                requested_players=("p1", "p2"),
                trajectory=BattleTrajectory(battle_id="search-play", format_id="gen3randombattle", seed=7),
            )
            self.assertEqual(getattr(search_policy.opponent_action_planner, "planner_id"), "checkpoint")
            self.assertIsNone(search_policy.opponent_action_scenario_planner)
            self.assertEqual(search_policy.opponent_action_planner(context, __import__("random").Random(1)), {"p2": 2})
            return FakeReport()

        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)) as load,
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25) as value_eval,
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8) as prior_eval,
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(0.1, 0.2, 0.7) + (0.0,) * 6) as opponent_eval,
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--games",
                    "3",
                    "--seed-start",
                    "99",
                    "--max-decision-rounds",
                    "12",
                    "--opponent-policy",
                    "random-legal",
                    "--cpuct",
                    "0.75",
                    "--leaf-rollout-rounds",
                    "2",
                    "--selection-mode",
                    "value",
                    "--min-value-improvement",
                    "0.2",
                    "--device",
                    "cpu",
                    "--temperature",
                    "1.5",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(Path("checkpoint.pt"), map_location="cpu")
        self.assertEqual(captured["games"], 3)
        self.assertEqual(captured["seed_start"], 99)
        self.assertEqual(captured["rollout_config"].max_decision_rounds, 12)
        matchups = tuple(captured["matchups"])
        self.assertEqual([matchup.label for matchup in matchups], [
            "neural-smoke vs random-legal",
            "random-legal vs neural-smoke",
            "neural-smoke+root-puct vs random-legal",
            "random-legal vs neural-smoke+root-puct",
        ])
        self.assertEqual(matchups[0].p1_policy.policy_id, "neural-smoke")
        self.assertEqual(matchups[2].p1_policy.policy_id, "neural-smoke+root-puct")
        self.assertEqual(matchups[3].p2_policy.policy_id, "neural-smoke+root-puct")
        self.assertEqual(matchups[2].p1_policy.cpuct, 0.75)
        self.assertEqual(matchups[2].p1_policy.selection_mode, "value")
        self.assertEqual(matchups[2].p1_policy.minimum_value_improvement, 0.2)
        self.assertEqual(matchups[2].p1_policy.leaf_rollout_decision_rounds, 2)
        self.assertIsNotNone(matchups[2].p1_policy.leaf_rollout_policy_factory)
        self.assertEqual(matchups[2].p1_policy.leaf_rollout_policy_factory("p1").policy_id, "neural-smoke+root-puct-leaf-p1")
        self.assertEqual(matchups[2].p1_policy.leaf_rollout_policy_factory("p2").policy_id, "neural-smoke+root-puct-leaf-p2")
        self.assertEqual(
            matchups[2].p1_policy.leaf_rollout_metadata,
            {"root_puct_leaf_rollout_opponent_policy": "checkpoint"},
        )
        self.assertTrue(matchups[2].p1_policy.allow_fallback)
        self.assertEqual(value_eval.call_args.kwargs["model"], fake_model)
        self.assertEqual(value_eval.call_args.kwargs["device"], "cpu")
        self.assertEqual(prior_eval.call_args.kwargs["temperature"], 1.5)
        self.assertEqual(opponent_eval.call_args.kwargs["temperature"], 1.5)
        self.assertEqual(json.loads(stdout.getvalue()), {"matchups": 4})

    def test_neural_cli_root_puct_play_benchmark_can_average_checkpoint_opponent_action_scenarios(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 4}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1))
        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            search_policy = tuple(kwargs["matchups"])[2].p1_policy
            context = PolicyContext(
                player_id="p1",
                decision_round_index=0,
                battle_id="search-play",
                format_id="gen3randombattle",
                seed=7,
                observation=observation(1),
                requested_players=("p1", "p2"),
                trajectory=BattleTrajectory(battle_id="search-play", format_id="gen3randombattle", seed=7),
                requested_legal_action_masks={
                    "p1": LEGAL_TWO_ACTION_MASK,
                    "p2": LEGAL_ACTION_ONE_TWO_MASK,
                },
            )
            scenarios = search_policy.opponent_action_scenario_planner(context, __import__("random").Random(1))
            self.assertEqual([dict(scenario.actions) for scenario in scenarios], [{"p2": 2}, {"p2": 1}])
            self.assertAlmostEqual(scenarios[0].weight, 0.7 / 0.9)
            self.assertAlmostEqual(scenarios[1].weight, 0.2 / 0.9)
            return FakeReport()

        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25),
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch(
                "pokezero.neural_cli.evaluate_transformer_opponent_action_priors",
                return_value=(0.1, 0.2, 0.7) + (0.0,) * 6,
            ),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--games",
                    "3",
                    "--opponent-policy",
                    "random-legal",
                    "--root-opponent-action-scenarios",
                    "2",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        matchups = tuple(captured["matchups"])
        search_policy = matchups[2].p1_policy
        self.assertEqual(getattr(search_policy.opponent_action_planner, "planner_id"), "checkpoint")
        self.assertIsNotNone(search_policy.opponent_action_scenario_planner)
        self.assertEqual(getattr(search_policy.opponent_action_scenario_planner, "planner_id"), "checkpoint-top2")
        self.assertEqual(json.loads(stdout.getvalue()), {"matchups": 4})

    def test_neural_cli_root_puct_play_benchmark_rejects_multi_scenarios_with_benchmark_root_opponent(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--root-opponent-action-policy",
                    "benchmark",
                    "--root-opponent-action-scenarios",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("root opponent action scenarios above one", stderr.getvalue())

    def test_neural_cli_root_puct_play_benchmark_can_use_benchmark_policy_for_root_opponent_actions(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 4}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1))
        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25),
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(0.1, 0.2, 0.7) + (0.0,) * 6),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--games",
                    "3",
                    "--opponent-policy",
                    "simple-legal",
                    "--root-opponent-action-policy",
                    "benchmark",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        matchups = tuple(captured["matchups"])
        p1_search = matchups[2].p1_policy
        p2_search = matchups[3].p2_policy
        self.assertEqual(getattr(p1_search.opponent_action_planner, "planner_id"), "benchmark")
        self.assertEqual(getattr(p2_search.opponent_action_planner, "planner_id"), "benchmark")
        p1_context = PolicyContext(
            player_id="p1",
            decision_round_index=0,
            battle_id="search-play",
            format_id="gen3randombattle",
            seed=7,
            observation=observation(1),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="search-play", format_id="gen3randombattle", seed=7),
            requested_legal_action_masks={
                "p1": LEGAL_TWO_ACTION_MASK,
                "p2": LEGAL_ACTION_ONE_MASK,
            },
            requested_observations={
                "p1": observation(1),
                "p2": observation(2, legal_action_mask=LEGAL_ACTION_ONE_MASK),
            },
        )
        p2_context = PolicyContext(
            player_id="p2",
            decision_round_index=0,
            battle_id="search-play",
            format_id="gen3randombattle",
            seed=7,
            observation=observation(2),
            requested_players=("p1", "p2"),
            trajectory=BattleTrajectory(battle_id="search-play", format_id="gen3randombattle", seed=7),
            requested_legal_action_masks={
                "p1": LEGAL_ACTION_ONE_MASK,
                "p2": LEGAL_TWO_ACTION_MASK,
            },
            requested_observations={
                "p1": observation(1, legal_action_mask=LEGAL_ACTION_ONE_MASK),
                "p2": observation(2),
            },
        )
        rng = __import__("random").Random(1)
        self.assertEqual(p1_search.opponent_action_planner(p1_context, rng), {"p2": 1})
        self.assertEqual(p2_search.opponent_action_planner(p2_context, rng), {"p1": 1})
        self.assertEqual(json.loads(stdout.getvalue()), {"matchups": 4})

    def test_neural_cli_root_puct_play_benchmark_can_use_benchmark_opponent_for_leaf_rollouts(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 4}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1))
        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25),
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(0.1, 0.2, 0.7) + (0.0,) * 6),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--games",
                    "3",
                    "--opponent-policy",
                    "simple-legal",
                    "--leaf-rollout-rounds",
                    "2",
                    "--leaf-rollout-opponent-policy",
                    "benchmark",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        matchups = tuple(captured["matchups"])
        p1_search = matchups[2].p1_policy
        p2_search = matchups[3].p2_policy
        self.assertEqual([matchup.label for matchup in matchups], [
            "neural-smoke vs simple-legal",
            "simple-legal vs neural-smoke",
            "neural-smoke+root-puct vs simple-legal",
            "simple-legal vs neural-smoke+root-puct",
        ])
        self.assertEqual(p1_search.leaf_rollout_decision_rounds, 2)
        self.assertEqual(p2_search.leaf_rollout_decision_rounds, 2)
        self.assertIsNotNone(p1_search.leaf_rollout_policy_factory)
        self.assertIsNotNone(p2_search.leaf_rollout_policy_factory)
        self.assertEqual(p1_search.leaf_rollout_policy_factory("p1").policy_id, "neural-smoke+root-puct-leaf-p1")
        p1_search_opponent = p1_search.leaf_rollout_policy_factory("p2")
        self.assertEqual(p1_search_opponent.policy_id, "simple-legal")
        self.assertIs(p1_search.leaf_rollout_policy_factory("p2"), p1_search_opponent)
        p2_search_opponent = p2_search.leaf_rollout_policy_factory("p1")
        self.assertEqual(p2_search_opponent.policy_id, "simple-legal")
        self.assertIs(p2_search.leaf_rollout_policy_factory("p1"), p2_search_opponent)
        self.assertEqual(p2_search.leaf_rollout_policy_factory("p2").policy_id, "neural-smoke+root-puct-leaf-p2")
        self.assertEqual(
            p1_search.leaf_rollout_metadata,
            {"root_puct_leaf_rollout_opponent_policy": "benchmark"},
        )
        self.assertEqual(
            p2_search.leaf_rollout_metadata,
            {"root_puct_leaf_rollout_opponent_policy": "benchmark"},
        )
        self.assertEqual(json.loads(stdout.getvalue()), {"matchups": 4})

    def test_neural_cli_root_puct_play_benchmark_can_sweep_leaf_depths(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 6}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1))
        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25),
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(0.1, 0.2, 0.7) + (0.0,) * 6),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--games",
                    "3",
                    "--opponent-policy",
                    "random-legal",
                    "--leaf-rollout-rounds-sweep",
                    "0",
                    "--leaf-rollout-rounds-sweep",
                    "2",
                    "--leaf-rollout-rounds-sweep",
                    "2",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        matchups = tuple(captured["matchups"])
        self.assertEqual([matchup.label for matchup in matchups], [
            "neural-smoke vs random-legal",
            "random-legal vs neural-smoke",
            "neural-smoke+root-puct-leaf0 vs random-legal",
            "random-legal vs neural-smoke+root-puct-leaf0",
            "neural-smoke+root-puct-leaf2 vs random-legal",
            "random-legal vs neural-smoke+root-puct-leaf2",
        ])
        leaf0 = matchups[2].p1_policy
        leaf2 = matchups[4].p1_policy
        self.assertEqual(leaf0.policy_id, "neural-smoke+root-puct-leaf0")
        self.assertEqual(leaf0.leaf_rollout_decision_rounds, 0)
        self.assertIsNone(leaf0.leaf_rollout_policy_factory)
        self.assertEqual(leaf2.policy_id, "neural-smoke+root-puct-leaf2")
        self.assertEqual(leaf2.leaf_rollout_decision_rounds, 2)
        self.assertIsNotNone(leaf2.leaf_rollout_policy_factory)
        self.assertEqual(
            leaf2.leaf_rollout_policy_factory("p2").policy_id,
            "neural-smoke+root-puct-leaf2-leaf-p2",
        )
        self.assertEqual(json.loads(stdout.getvalue()), {"matchups": 6})

    def test_neural_cli_root_puct_play_benchmark_tags_single_sweep_depth(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 4}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1))
        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25),
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(0.1, 0.2, 0.7) + (0.0,) * 6),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--games",
                    "3",
                    "--opponent-policy",
                    "random-legal",
                    "--leaf-rollout-rounds-sweep",
                    "2",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        matchups = tuple(captured["matchups"])
        self.assertEqual([matchup.label for matchup in matchups], [
            "neural-smoke vs random-legal",
            "random-legal vs neural-smoke",
            "neural-smoke+root-puct-leaf2 vs random-legal",
            "random-legal vs neural-smoke+root-puct-leaf2",
        ])
        self.assertEqual(matchups[2].p1_policy.policy_id, "neural-smoke+root-puct-leaf2")
        self.assertEqual(matchups[2].p1_policy.leaf_rollout_decision_rounds, 2)
        self.assertEqual(json.loads(stdout.getvalue()), {"matchups": 4})

    def test_neural_cli_root_puct_benchmark_wires_checkpoint_callbacks_and_source_policies(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"evaluated_prefixes": 2}

        fake_model = object()
        fake_training_result = object()
        captured = {}

        def fake_benchmark_root_puct_search(**kwargs):
            captured.update(kwargs)
            self.assertEqual(kwargs["value_fn"]((observation(1),)), 0.25)
            self.assertEqual(kwargs["prior_fn"]((observation(1),)), (1.0,) + (0.0,) * 8)
            return FakeReport()

        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)) as load,
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25) as value_eval,
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8) as prior_eval,
            patch("pokezero.neural_cli.benchmark_root_puct_search", side_effect=fake_benchmark_root_puct_search),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--games",
                    "3",
                    "--prefixes-per-game",
                    "4",
                    "--seed-start",
                    "99",
                    "--max-decision-rounds",
                    "12",
                    "--p1-policy",
                    "random-legal",
                    "--p2-policy",
                    "simple-legal",
                    "--search-player",
                    "p2",
                    "--cpuct",
                    "0.75",
                    "--device",
                    "cpu",
                    "--temperature",
                    "1.5",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(Path("checkpoint.pt"), map_location="cpu")
        self.assertEqual(captured["games"], 3)
        self.assertEqual(captured["prefixes_per_game"], 4)
        self.assertEqual(captured["seed_start"], 99)
        self.assertEqual(captured["search_player"], "p2")
        self.assertEqual(captured["cpuct"], 0.75)
        self.assertEqual(captured["rollout_config"].max_decision_rounds, 12)
        self.assertEqual(captured["policies"]["p1"].policy_id, "random-legal")
        self.assertEqual(captured["policies"]["p2"].policy_id, "simple-legal")
        self.assertEqual(value_eval.call_args.kwargs["model"], fake_model)
        self.assertEqual(value_eval.call_args.kwargs["result"], fake_training_result)
        self.assertEqual(value_eval.call_args.kwargs["device"], "cpu")
        self.assertEqual(prior_eval.call_args.kwargs["temperature"], 1.5)
        self.assertEqual(json.loads(stdout.getvalue()), {"evaluated_prefixes": 2})

    def test_neural_cli_root_puct_counterfactual_wires_continuation_policies(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"average_rollout_value_delta": 0.5}

        fake_model = object()
        fake_training_result = object()
        captured = {}

        def fake_benchmark_root_puct_counterfactual_rollouts(**kwargs):
            captured.update(kwargs)
            self.assertEqual(kwargs["value_fn"]((observation(1),)), 0.25)
            self.assertEqual(kwargs["prior_fn"]((observation(1),)), (1.0,) + (0.0,) * 8)
            return FakeReport()

        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)) as load,
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25) as value_eval,
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8) as prior_eval,
            patch(
                "pokezero.neural_cli.benchmark_root_puct_counterfactual_rollouts",
                side_effect=fake_benchmark_root_puct_counterfactual_rollouts,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-counterfactual",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--games",
                    "3",
                    "--prefixes-per-game",
                    "4",
                    "--seed-start",
                    "99",
                    "--max-decision-rounds",
                    "12",
                    "--p1-policy",
                    "random-legal",
                    "--p2-policy",
                    "simple-legal",
                    "--continuation-p1-policy",
                    "simple-legal",
                    "--continuation-p2-policy",
                    "random-legal",
                    "--search-player",
                    "p2",
                    "--cpuct",
                    "0.75",
                    "--device",
                    "cpu",
                    "--temperature",
                    "1.5",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(Path("checkpoint.pt"), map_location="cpu")
        self.assertEqual(captured["games"], 3)
        self.assertEqual(captured["prefixes_per_game"], 4)
        self.assertEqual(captured["seed_start"], 99)
        self.assertEqual(captured["search_player"], "p2")
        self.assertEqual(captured["cpuct"], 0.75)
        self.assertEqual(captured["rollout_config"].max_decision_rounds, 12)
        self.assertEqual(captured["policies"]["p1"].policy_id, "random-legal")
        self.assertEqual(captured["policies"]["p2"].policy_id, "simple-legal")
        self.assertEqual(captured["continuation_policies"]["p1"].policy_id, "simple-legal")
        self.assertEqual(captured["continuation_policies"]["p2"].policy_id, "random-legal")
        self.assertEqual(value_eval.call_args.kwargs["model"], fake_model)
        self.assertEqual(value_eval.call_args.kwargs["result"], fake_training_result)
        self.assertEqual(value_eval.call_args.kwargs["device"], "cpu")
        self.assertEqual(prior_eval.call_args.kwargs["temperature"], 1.5)
        self.assertEqual(json.loads(stdout.getvalue()), {"average_rollout_value_delta": 0.5})

    def test_neural_cli_value_calibration_wires_checkpoint_and_data(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"examples": 3, "mse": 0.25}

        fake_model = object()
        fake_training_result = object()
        stdout = io.StringIO()

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)) as load,
            patch("pokezero.neural_cli.evaluate_value_calibration", return_value=FakeReport()) as evaluate,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts-a.jsonl",
                    "rollouts-b.jsonl",
                    "--batch-size",
                    "7",
                    "--bins",
                    "5",
                    "--device",
                    "cpu",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(Path("checkpoint.pt"), map_location="cpu")
        self.assertEqual(evaluate.call_args.kwargs["model"], fake_model)
        self.assertEqual(evaluate.call_args.kwargs["training_result"], fake_training_result)
        self.assertEqual(evaluate.call_args.kwargs["paths"], [Path("rollouts-a.jsonl"), Path("rollouts-b.jsonl")])
        self.assertEqual(evaluate.call_args.kwargs["batch_size"], 7)
        self.assertEqual(evaluate.call_args.kwargs["bins"], 5)
        self.assertEqual(evaluate.call_args.kwargs["device"], "cpu")
        self.assertEqual(json.loads(stdout.getvalue()), {"examples": 3, "mse": 0.25})

    def test_neural_cli_value_calibration_resolves_default_device(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        class FakeReport:
            def to_dict(self) -> dict:
                return {"examples": 3}

        fake_model = object()
        fake_training_result = object()

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)) as load,
            patch("pokezero.neural_cli.evaluate_value_calibration", return_value=FakeReport()) as evaluate,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(Path("checkpoint.pt"), map_location="cuda")
        self.assertEqual(evaluate.call_args.kwargs["device"], "cuda")

    def test_neural_cli_train_can_write_value_calibration_artifact(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"examples": 4, "sign_accuracy": 0.75}

        def fake_train(paths, *, model_config, training_config, initial_model=None):
            return object(), TransformerTrainingResult(
                model_config=model_config,
                training_config=training_config,
                epochs=(
                    TransformerEpochMetrics(
                        epoch=1,
                        examples=4,
                        loss=0.25,
                        policy_loss=0.2,
                        policy_accuracy=0.75,
                    ),
                ),
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            calibration_path = Path(temp_dir) / "value-calibration.json"
            stdout = io.StringIO()
            with (
                patch("pokezero.neural_cli.train_transformer_policy", side_effect=fake_train) as train,
                patch("pokezero.neural_cli.save_transformer_checkpoint") as save,
                patch("pokezero.neural_cli.evaluate_value_calibration", return_value=FakeReport()) as evaluate,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = neural_cli_main(
                    [
                        "train",
                        "--data",
                        "train-rollouts.jsonl",
                        "--out",
                        str(checkpoint_path),
                        "--showdown-root",
                        "/tmp/showdown",
                        "--value-calibration-data",
                        "calibration-rollouts.jsonl",
                        "--value-calibration-out",
                        str(calibration_path),
                        "--value-calibration-batch-size",
                        "9",
                        "--value-calibration-bins",
                        "6",
                    ]
                )

            payload = json.loads(calibration_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(train.call_args.args[0], [Path("train-rollouts.jsonl")])
        self.assertEqual(save.call_args.args[0], checkpoint_path)
        self.assertEqual(evaluate.call_args.kwargs["paths"], [Path("calibration-rollouts.jsonl")])
        self.assertEqual(evaluate.call_args.kwargs["batch_size"], 9)
        self.assertEqual(evaluate.call_args.kwargs["bins"], 6)
        self.assertEqual(payload["paths"], ["calibration-rollouts.jsonl"])
        self.assertEqual(payload["report"]["sign_accuracy"], 0.75)
        self.assertIn(f"value_calibration: {calibration_path}", stdout.getvalue())

    def test_neural_cli_train_can_warm_start_value_only_finetune(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        fake_model = object()
        fake_model_config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",),
            category_oov_buckets=2,
            policy_id="base-policy",
        )
        fake_loaded_result = TransformerTrainingResult(
            model_config=fake_model_config,
            training_config=TransformerTrainingConfig(),
            epochs=(),
        )

        def fake_train(paths, *, model_config, training_config, initial_model=None):
            return object(), TransformerTrainingResult(
                model_config=model_config,
                training_config=training_config,
                epochs=(
                    TransformerEpochMetrics(
                        epoch=1,
                        examples=4,
                        loss=0.25,
                        policy_loss=0.0,
                        policy_accuracy=0.5,
                        value_loss=0.25,
                    ),
                ),
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            with (
                patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_loaded_result)) as load,
                patch("pokezero.neural_cli.train_transformer_policy", side_effect=fake_train) as train,
                patch("pokezero.neural_cli.save_transformer_checkpoint") as save,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = neural_cli_main(
                    [
                        "train",
                        "--data",
                        "train-rollouts.jsonl",
                        "--out",
                        str(checkpoint_path),
                        "--initial-checkpoint",
                        "base.pt",
                        "--policy-id",
                        "value-finetuned",
                        "--objective",
                        "value-only",
                        "--freeze-non-value-parameters",
                        "--device",
                        "cpu",
                    ]
                )

        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(Path("base.pt"), map_location="cpu")
        self.assertEqual(train.call_args.kwargs["initial_model"], fake_model)
        self.assertEqual(train.call_args.kwargs["model_config"].policy_id, "value-finetuned")
        self.assertEqual(train.call_args.kwargs["training_config"].objective, "value-only")
        self.assertTrue(train.call_args.kwargs["training_config"].freeze_non_value_parameters)
        self.assertEqual(save.call_args.args[0], checkpoint_path)

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
                    "--switch-action-loss-weight",
                    "1.5",
                    "--action-family-loss-weight",
                    "0.75",
                    "--switch-target-loss-weight",
                    "0.5",
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
                    "--value-calibration",
                    "--value-calibration-scope",
                    "history",
                    "--value-calibration-batch-size",
                    "9",
                    "--value-calibration-bins",
                    "6",
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
        self.assertEqual(kwargs["training_config"].switch_action_loss_weight, 1.5)
        self.assertEqual(kwargs["training_config"].action_family_loss_weight, 0.75)
        self.assertEqual(kwargs["training_config"].switch_target_loss_weight, 0.5)
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
        self.assertEqual(kwargs["value_calibration_config"].scope, "history")
        self.assertEqual(kwargs["value_calibration_config"].batch_size, 9)
        self.assertEqual(kwargs["value_calibration_config"].bins, 6)

    @staticmethod
    def _fake_iterate_result():
        fake_epoch = type("FakeEpoch", (), {"loss": 0.25, "policy_accuracy": 0.75})()
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
        return type(
            "FakeResult",
            (),
            {
                "run_dir": Path("run"),
                "iterations": (fake_iteration,),
                "latest_checkpoint_path": Path("run/iteration-0001/transformer-policy.pt"),
                "to_dict": lambda self: {"ok": True},
            },
        )()

    def _run_iterate_capturing_model_config(self, extra_args: list[str]) -> Any:
        captured: dict[str, Any] = {}

        def _capture(**kwargs):
            captured["model_config"] = kwargs["model_config"]
            return self._fake_iterate_result()

        with (
            patch("pokezero.neural_cli.run_neural_selfplay_iterations", side_effect=_capture),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                ["iterate", "--run-dir", "run", "--iterations", "1", "--games-per-iteration", "2",
                 "--initial-policy", "simple-legal"] + extra_args
            )
        self.assertEqual(exit_code, 0)
        return captured["model_config"]

    def test_neural_cli_iterate_builds_compact_randbat_dex_config(self) -> None:
        # setUp stubs gen3_category_vocabulary to a 3-token string vocab (oov_buckets=16).
        model_config = self._run_iterate_capturing_model_config(
            ["--showdown-root", "/tmp/showdown", "--category-oov-buckets", "4"]
        )
        self.assertEqual(model_config.category_vocab, ("move:c", "species:a", "species:b"))
        self.assertEqual(model_config.categorical_vocab_size, 1 + 3 + 16)

    def test_neural_cli_iterate_requires_showdown_root(self) -> None:
        with (
            patch("pokezero.neural_cli.run_neural_selfplay_iterations") as run,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                ["iterate", "--run-dir", "run", "--iterations", "1", "--games-per-iteration", "2",
                 "--initial-policy", "simple-legal"]
            )
        self.assertEqual(exit_code, 1)
        run.assert_not_called()

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
                    "--showdown-root",
                    "/tmp/showdown",
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
                        "--showdown-root",
                        "/tmp/showdown",
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
                    "--showdown-root",
                    "/tmp/showdown",
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
                model_config=TransformerPolicyConfig.compact_category(
                    category_vocab=tuple(range(1, 17)),
                    category_oov_buckets=4,
                    policy_id="neural-smoke",
                    window_size=2,
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
            opponent_priors = evaluate_transformer_opponent_action_priors(
                model=restored_model,
                result=restored_result,
                observations=(observation(1),),
                device="cpu",
            )
            restored_model.eval()
            self.assertFalse(restored_model.training)
            _, continued_result = train_transformer_policy(
                data_path,
                model_config=TransformerPolicyConfig.compact_category(
                    category_vocab=tuple(range(1, 17)),
                    category_oov_buckets=4,
                    policy_id="neural-smoke-continued",
                    window_size=2,
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
            self.assertTrue(restored_model.training)

        self.assertEqual(result.final_metrics.examples, 2)
        self.assertEqual(continued_result.model_config.policy_id, "neural-smoke-continued")
        self.assertIn(decision.action_index, {0, 1})
        self.assertEqual(policy.policy_id, "neural-smoke")
        self.assertEqual(len(opponent_priors), 9)
        self.assertAlmostEqual(sum(opponent_priors), 1.0, places=6)

    def test_value_only_freeze_updates_value_head_only(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            model_config = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)),
                category_oov_buckets=4,
                policy_id="value-finetune-smoke",
                window_size=2,
                token_type_vocab_size=8,
                categorical_feature_count=1,
                numeric_feature_count=1,
                embedding_dim=16,
                transformer_layers=1,
                attention_heads=4,
                feedforward_dim=32,
                dropout=0.0,
            )
            model = EntityTokenTransformerPolicy(model_config)
            before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

            trained_model, result = train_transformer_policy(
                data_path,
                model_config=model_config,
                training_config=TransformerTrainingConfig(
                    batch_size=2,
                    epochs=1,
                    window_size=2,
                    max_batches=1,
                    device="cpu",
                    objective="value-only",
                    freeze_non_value_parameters=True,
                ),
                initial_model=model,
            )

            changed = {
                name
                for name, parameter in trained_model.named_parameters()
                if not require_torch().equal(before[name], parameter.detach())
            }

        self.assertEqual(result.final_metrics.policy_loss, 0.0)
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("value_head.") for name in changed))


if __name__ == "__main__":
    unittest.main()
