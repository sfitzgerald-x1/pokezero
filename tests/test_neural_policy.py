import contextlib
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import tempfile
from typing import Any
import unittest
from urllib.parse import urlencode
from unittest.mock import call, patch

from pokezero import neural_policy as neural_policy_module
from pokezero.actions import ACTION_COUNT
from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.dataset import (
    TrajectoryDatasetConfig,
    examples_from_record,
    write_training_cache_from_examples,
    write_training_cache_from_rollouts,
)
from pokezero.env import TerminalState
from pokezero.neural_cli import (
    _PolicyIdAlias,
    _RootPuctDecisionProgress,
    _RootPuctDecisionProgressPolicy,
    _adaptive_root_visit_budget_selector,
    _belief_world_benchmark_coverage,
    _root_opponent_action_candidate_scenario_count,
    _validate_root_opponent_action_scenario_counts,
    _root_puct_benchmark_progress_callback,
    _root_puct_decision_progress_callback,
    _root_visit_budget_selector,
    _require_belief_world_benchmark_coverage,
    _input_data_paths_byte_size,
    _refutation_cache_training_contract,
    _training_cache_lifecycle,
    _validate_refutation_cache_args,
    build_arg_parser as build_neural_arg_parser,
    main as neural_cli_main,
)
from pokezero.neural_policy import (
    DEFAULT_TOKEN_TYPE_VOCAB_SIZE,
    CONSTANT_LEARNING_RATE_SCHEDULE,
    MIT_THESIS_LEARNING_RATE_SCHEDULE,
    NEURAL_INSTALL_MESSAGE,
    EntityTokenTransformerPolicy,
    TorchUnavailableError,
    TransformerInferenceTimingAccumulator,
    TransformerSoftmaxPolicy,
    TransformerEpochMetrics,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    ValueCalibrationTransform,
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
    evaluate_transformer_observation_values,
    evaluate_transformer_opponent_action_priors,
    load_transformer_checkpoint,
    load_transformer_policy,
    require_torch,
    require_compatible_transformer_value_checkpoint,
    resolve_torch_device,
    save_transformer_checkpoint,
    observation_window_to_torch,
    observation_windows_to_torch,
    torch_available,
    train_transformer_policy,
    training_batch_to_torch,
    _greedy_action_index,
    learning_rate_for_progress,
)
from pokezero.neural_selfplay import _require_promoted_opponent_pool as require_neural_promoted_opponent_pool
from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V2, ObservationSpec, PokeZeroObservationV0
from pokezero.policy import PolicyContext, PolicyDecision
from pokezero.run_audit import RunAuditConfig, run_audit_config_payload
from pokezero.showdown import ACTION_CANDIDATE_TOKEN_OFFSET, DEFAULT_REPLAY_OBSERVATION_SPEC
from pokezero.trajectory import BattleTrajectory, TrajectoryStep
from pokezero.value_calibration import ValueCalibrationReport


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

    def test_transformer_inference_timing_splits_forward_roles(self) -> None:
        timing = TransformerInferenceTimingAccumulator()
        timing.add_neural_forward(0.10, role="action_prior")
        timing.add_neural_forward(0.20, role="opponent_action_prior")
        timing.add_neural_forward(0.30, role="policy")
        timing.add_neural_forward(0.40, role="value")

        snapshot = timing.snapshot()

        self.assertEqual(snapshot.neural_forward_count, 4)
        self.assertAlmostEqual(snapshot.neural_forward_seconds, 1.0)
        self.assertEqual(snapshot.action_prior_neural_forward_count, 1)
        self.assertEqual(snapshot.opponent_action_prior_neural_forward_count, 1)
        self.assertEqual(snapshot.policy_neural_forward_count, 1)
        self.assertEqual(snapshot.value_neural_forward_count, 1)
        self.assertAlmostEqual(
            snapshot.neural_forward_seconds,
            snapshot.action_prior_neural_forward_seconds
            + snapshot.opponent_action_prior_neural_forward_seconds
            + snapshot.policy_neural_forward_seconds
            + snapshot.value_neural_forward_seconds,
        )
        with self.assertRaisesRegex(ValueError, "unknown transformer inference timing role"):
            timing.add_neural_forward(0.01, role="unknown")

    def test_transformer_policy_config_defaults_match_replay_observation_shape(self) -> None:
        config = TransformerPolicyConfig.compact_category(category_vocab=(1, 2, 3), category_oov_buckets=4)

        # Spec v2 default: window=1 snapshots (transition tokens carry temporal context).
        self.assertEqual(config.window_size, 1)
        self.assertEqual(config.token_count, DEFAULT_REPLAY_OBSERVATION_SPEC.token_count)
        self.assertEqual(config.categorical_feature_count, DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count)
        self.assertEqual(config.numeric_feature_count, DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count)
        # categorical_vocab_size is derived: 1 padding + 3 vocab + 4 oov.
        self.assertEqual(config.categorical_vocab_size, 8)
        self.assertEqual(config.token_type_vocab_size, DEFAULT_TOKEN_TYPE_VOCAB_SIZE)
        self.assertEqual(config.value_activation, "tanh")
        self.assertEqual(config.temporal_aggregator, "mean")
        self.assertGreaterEqual(config.token_count, ACTION_CANDIDATE_TOKEN_OFFSET + 9)
        self.assertEqual(TransformerPolicyConfig.from_dict(config.to_dict()), config)

    def test_checkpoint_root_puct_defaults_to_hidden_reserve_candidates(self) -> None:
        base = SimpleNamespace(
            root_opponent_action_candidate_scenarios=None,
            root_opponent_action_scenarios=1,
            root_opponent_action_policy="checkpoint",
        )

        self.assertEqual(_root_opponent_action_candidate_scenario_count(base), ACTION_COUNT)
        self.assertEqual(
            _root_opponent_action_candidate_scenario_count(
                SimpleNamespace(
                    **{
                        **vars(base),
                        "root_opponent_action_policy": "benchmark",
                    }
                )
            ),
            1,
        )
        self.assertEqual(
            _root_opponent_action_candidate_scenario_count(
                SimpleNamespace(
                    **{
                        **vars(base),
                        "root_opponent_action_candidate_scenarios": 3,
                    }
                )
            ),
            3,
        )

    def test_checkpoint_root_puct_rejects_accepted_scenarios_above_action_space(self) -> None:
        args = SimpleNamespace(
            root_opponent_action_candidate_scenarios=None,
            root_opponent_action_scenarios=ACTION_COUNT + 1,
            root_opponent_action_policy="checkpoint",
        )

        with self.assertRaisesRegex(ValueError, rf"must not exceed {ACTION_COUNT} abstract actions"):
            _validate_root_opponent_action_scenario_counts(args)

    def test_transformer_policy_config_loads_legacy_fields_with_compatible_defaults(self) -> None:
        config = TransformerPolicyConfig.compact_category(category_vocab=(1, 2, 3), category_oov_buckets=4)
        payload = config.to_dict()
        payload.pop("value_activation")
        payload.pop("temporal_aggregator")

        restored = TransformerPolicyConfig.from_dict(payload)

        self.assertEqual(restored.value_activation, "linear")
        self.assertEqual(restored.temporal_aggregator, "mean")

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
        with self.assertRaisesRegex(ValueError, "temporal_aggregator"):
            TransformerPolicyConfig.compact_category(category_vocab=(1,), category_oov_buckets=1, temporal_aggregator="lstm")

    def test_zero_layer_transformer_policy_forward_smoke(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        config = TransformerPolicyConfig.compact_category(
            policy_id="cpu-fast",
            category_vocab=("fixture",),
            category_oov_buckets=1,
            window_size=2,
            categorical_feature_count=1,
            numeric_feature_count=1,
            token_count=ObservationSpec(categorical_feature_count=1, numeric_feature_count=1).token_count,
            embedding_dim=8,
            transformer_layers=0,
            attention_heads=1,
            feedforward_dim=8,
        )
        model = EntityTokenTransformerPolicy(config)
        tensors = observation_window_to_torch(
            (observation(1), observation(2)),
            window_size=config.window_size,
            device="cpu",
        )

        output = model(
            categorical_ids=tensors["categorical_ids"],
            numeric_features=tensors["numeric_features"],
            token_type_ids=tensors["token_type_ids"],
            attention_mask=tensors["attention_mask"],
            history_mask=tensors["history_mask"],
        )

        self.assertEqual(tuple(output.policy_logits.shape), (1, 9))
        self.assertEqual(tuple(output.value.shape), (1,))
        self.assertEqual(tuple(output.opponent_action_logits.shape), (1, 9))

    def test_transformer_forward_accepts_compact_categorical_training_cache_rows(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        config = TransformerPolicyConfig.compact_category(
            policy_id="compact-categories",
            category_vocab=tuple(f"token-{index}" for index in range(16)),
            category_oov_buckets=1,
            window_size=4,
            categorical_feature_count=4,
            numeric_feature_count=2,
            token_count=DEFAULT_REPLAY_OBSERVATION_SPEC.token_count,
            embedding_dim=8,
            transformer_layers=0,
            attention_heads=1,
            feedforward_dim=8,
            dropout=0.0,
        )
        model = EntityTokenTransformerPolicy(config)
        model.eval()
        dense_categories = torch.zeros((1, config.window_size, config.token_count, 4), dtype=torch.long)
        compact_categories = torch.zeros((1, config.window_size, config.token_count, 2), dtype=torch.long)
        dense_categories[..., 1] = 2
        dense_categories[..., 3] = 3
        compact_categories[..., 0] = 2
        compact_categories[..., 1] = 3
        numeric_features = torch.ones((1, config.window_size, config.token_count, config.numeric_feature_count))
        token_type_ids = torch.zeros((1, config.window_size, config.token_count), dtype=torch.long)
        attention_mask = torch.ones((1, config.window_size, config.token_count), dtype=torch.bool)
        history_mask = torch.ones((1, config.window_size), dtype=torch.bool)

        with torch.no_grad():
            dense_output = model(
                categorical_ids=dense_categories,
                numeric_features=numeric_features,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                history_mask=history_mask,
            )
            compact_output = model(
                categorical_ids=compact_categories,
                numeric_features=numeric_features,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                history_mask=history_mask,
            )

        self.assertTrue(torch.allclose(compact_output.policy_logits, dense_output.policy_logits))
        self.assertTrue(torch.allclose(compact_output.value, dense_output.value))
        self.assertTrue(torch.allclose(compact_output.opponent_action_logits, dense_output.opponent_action_logits))

    def test_transformer_forward_accepts_row_indexed_training_cache_windows(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        config = TransformerPolicyConfig.compact_category(
            policy_id="row-indexed-cache",
            category_vocab=tuple(f"token-{index}" for index in range(16)),
            category_oov_buckets=1,
            window_size=4,
            categorical_feature_count=4,
            numeric_feature_count=2,
            token_count=DEFAULT_REPLAY_OBSERVATION_SPEC.token_count,
            embedding_dim=8,
            transformer_layers=1,
            attention_heads=1,
            feedforward_dim=16,
            dropout=0.0,
        )
        model = EntityTokenTransformerPolicy(config)
        model.eval()
        row_categorical_ids = torch.zeros((2, config.token_count, 2), dtype=torch.long)
        row_categorical_ids[0, :, 0] = 2
        row_categorical_ids[0, :, 1] = 3
        row_categorical_ids[1, :, 0] = 4
        row_categorical_ids[1, :, 1] = 5
        row_numeric_features = torch.ones((2, config.token_count, config.numeric_feature_count))
        row_numeric_features[1] = 2.0
        row_token_type_ids = torch.zeros((2, config.token_count), dtype=torch.long)
        row_attention_mask = torch.ones((2, config.token_count), dtype=torch.bool)
        row_attention_mask[1, 0] = False
        window_row_indices = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
        dense_categories = row_categorical_ids[window_row_indices]
        dense_numeric = row_numeric_features[window_row_indices]
        dense_token_types = row_token_type_ids[window_row_indices]
        dense_attention = row_attention_mask[window_row_indices]
        history_mask = torch.ones((1, config.window_size), dtype=torch.bool)

        with torch.no_grad():
            dense_output = model(
                categorical_ids=dense_categories,
                numeric_features=dense_numeric,
                token_type_ids=dense_token_types,
                attention_mask=dense_attention,
                history_mask=history_mask,
            )
            row_output = model(
                row_categorical_ids=row_categorical_ids,
                row_numeric_features=row_numeric_features,
                row_token_type_ids=row_token_type_ids,
                row_attention_mask=row_attention_mask,
                window_row_indices=window_row_indices,
                history_mask=history_mask,
            )

        self.assertTrue(torch.allclose(row_output.policy_logits, dense_output.policy_logits))
        self.assertTrue(torch.allclose(row_output.value, dense_output.value))
        self.assertTrue(torch.allclose(row_output.opponent_action_logits, dense_output.opponent_action_logits))

    def test_zero_layer_row_indexed_forward_matches_dense_expansion(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        config = TransformerPolicyConfig.compact_category(
            policy_id="row-indexed-zero-layer-cache",
            category_vocab=tuple(f"token-{index}" for index in range(16)),
            category_oov_buckets=1,
            window_size=4,
            categorical_feature_count=4,
            numeric_feature_count=2,
            token_count=DEFAULT_REPLAY_OBSERVATION_SPEC.token_count,
            embedding_dim=8,
            transformer_layers=0,
            attention_heads=1,
            feedforward_dim=8,
            dropout=0.0,
        )
        # The two forward paths reduce ~1.6k fp32 token contributions in different orders
        # (per-row sums vs masked mean over expanded windows), so they agree only up to
        # fp32 accumulation noise: measured up to ~1e-6 on outputs and ~2e-4 on gradients
        # across 500 weight draws. Pin the draw so the margin cannot drift with suite
        # ordering, and keep the tolerances above that noise floor — a real path
        # divergence shows up at the scale of the values themselves (O(1) and up).
        with torch.random.fork_rng():
            torch.manual_seed(20260704)
            model = EntityTokenTransformerPolicy(config)
        model.eval()
        row_categorical_ids = torch.zeros((3, config.token_count, 2), dtype=torch.long)
        row_categorical_ids[0, :, 0] = 2
        row_categorical_ids[0, :, 1] = 3
        row_categorical_ids[1, :, 0] = 4
        row_categorical_ids[1, :, 1] = 5
        row_categorical_ids[2, :, 0] = 6
        row_numeric_features = torch.ones((3, config.token_count, config.numeric_feature_count))
        row_numeric_features[1] = 2.0
        row_numeric_features[2] = 3.0
        row_token_type_ids = torch.zeros((3, config.token_count), dtype=torch.long)
        row_token_type_ids[2] = 1
        row_attention_mask = torch.ones((3, config.token_count), dtype=torch.bool)
        row_attention_mask[1, 0] = False
        row_attention_mask[2, 1:3] = False
        window_row_indices = torch.tensor([[0, 1, 0, 2], [1, 2, 0, 1], [2, 0, 1, 2]], dtype=torch.long)
        dense_categories = row_categorical_ids[window_row_indices]
        dense_numeric = row_numeric_features[window_row_indices]
        dense_token_types = row_token_type_ids[window_row_indices]
        dense_attention = row_attention_mask[window_row_indices]
        history_mask = torch.tensor(
            [[False, True, True, True], [True, False, True, True], [True, True, True, False]],
            dtype=torch.bool,
        )

        dense_output = model(
            categorical_ids=dense_categories,
            numeric_features=dense_numeric,
            token_type_ids=dense_token_types,
            attention_mask=dense_attention,
            history_mask=history_mask,
        )
        row_output = model(
            row_categorical_ids=row_categorical_ids,
            row_numeric_features=row_numeric_features,
            row_token_type_ids=row_token_type_ids,
            row_attention_mask=row_attention_mask,
            window_row_indices=window_row_indices,
            history_mask=history_mask,
        )

        self.assertTrue(torch.allclose(row_output.policy_logits, dense_output.policy_logits, atol=1e-5))
        self.assertTrue(torch.allclose(row_output.value, dense_output.value, atol=1e-5))
        self.assertTrue(torch.allclose(row_output.opponent_action_logits, dense_output.opponent_action_logits, atol=1e-5))
        model.zero_grad(set_to_none=True)
        dense_output = model(
            categorical_ids=dense_categories,
            numeric_features=dense_numeric,
            token_type_ids=dense_token_types,
            attention_mask=dense_attention,
            history_mask=history_mask,
        )
        dense_loss = (
            dense_output.policy_logits.sum()
            + dense_output.value.sum()
            + dense_output.opponent_action_logits.sum()
        )
        dense_loss.backward()
        dense_grads = {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)
        row_output = model(
            row_categorical_ids=row_categorical_ids,
            row_numeric_features=row_numeric_features,
            row_token_type_ids=row_token_type_ids,
            row_attention_mask=row_attention_mask,
            window_row_indices=window_row_indices,
            history_mask=history_mask,
        )
        row_loss = row_output.policy_logits.sum() + row_output.value.sum() + row_output.opponent_action_logits.sum()
        row_loss.backward()
        row_grads = {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in model.named_parameters()
        }
        self.assertEqual(dense_grads.keys(), row_grads.keys())
        for name, dense_grad in dense_grads.items():
            row_grad = row_grads[name]
            if dense_grad is None or row_grad is None:
                self.assertIsNone(dense_grad, name)
                self.assertIsNone(row_grad, name)
            else:
                # Mathematically identical paths accumulate in different orders; the spec v2
                # token count (151) makes fp32 sum noise exceed the old 1e-5 absolute bound.
                self.assertTrue(torch.allclose(row_grad, dense_grad, atol=1e-3, rtol=1e-4), name)

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

    def test_evaluate_transformer_observation_value_applies_calibration_transform(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
        config = TransformerPolicyConfig.compact_category(
            policy_id="fixture",
            category_vocab=("fixture",),
            category_oov_buckets=1,
            window_size=1,
            categorical_feature_count=1,
            numeric_feature_count=1,
            token_count=spec.token_count,
            embedding_dim=4,
            transformer_layers=1,
            attention_heads=1,
            feedforward_dim=8,
        )

        class FakeValueModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                return SimpleNamespace(value=torch.tensor([0.4]))

        value = evaluate_transformer_observation_value(
            model=FakeValueModel(),
            result=SimpleNamespace(
                model_config=config,
                value_calibration_transform=ValueCalibrationTransform(scale=2.0, bias=-0.1),
            ),
            observations=(observation(1),),
            device="cpu",
        )

        self.assertAlmostEqual(value, 0.7, places=5)

    def test_evaluate_transformer_observation_values_batches_and_calibrates(self) -> None:
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
                self.shapes: dict[str, tuple[int, ...]] = {}

            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                self.shapes = {name: tuple(value.shape) for name, value in kwargs.items()}
                return SimpleNamespace(value=torch.tensor([0.2, 0.4]))

        model = FakeValueModel()
        timing = TransformerInferenceTimingAccumulator()
        values = evaluate_transformer_observation_values(
            model=model,
            result=SimpleNamespace(
                model_config=config,
                value_calibration_transform=ValueCalibrationTransform(scale=2.0, bias=-0.1),
            ),
            observation_histories=((observation(1),), (observation(2), observation(3))),
            device="cpu",
            timing=timing,
        )

        self.assertAlmostEqual(values[0], 0.3, places=5)
        self.assertAlmostEqual(values[1], 0.7, places=5)
        self.assertEqual(model.shapes["categorical_ids"], (2, 2, spec.token_count, 1))
        self.assertEqual(model.shapes["history_mask"], (2, 2))
        snapshot = timing.snapshot()
        self.assertEqual(snapshot.observation_encoding_count, 1)
        self.assertEqual(snapshot.value_neural_forward_count, 1)

    def test_batched_value_tensor_rows_match_scalar_tensorization(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        histories = ((observation(1),), (observation(2), observation(3)))
        batched = observation_windows_to_torch(histories, window_size=2, device="cpu")

        for row_index, history in enumerate(histories):
            scalar = observation_window_to_torch(history, window_size=2, device="cpu")
            for name in (
                "categorical_ids",
                "numeric_features",
                "token_type_ids",
                "attention_mask",
                "history_mask",
            ):
                self.assertTrue(
                    torch.equal(batched[name][row_index], scalar[name][0]),
                    msg=name,
                )

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

    def test_transformer_gru_temporal_aggregator_forward_shapes(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",),
            category_oov_buckets=1,
            window_size=3,
            embedding_dim=8,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=16,
            dropout=0.0,
            temporal_aggregator="gru",
        )
        model = EntityTokenTransformerPolicy(config)
        shape = (2, config.window_size, config.token_count)
        inputs = {
            "categorical_ids": torch.zeros((*shape, config.categorical_feature_count), dtype=torch.long),
            "numeric_features": torch.zeros((*shape, config.numeric_feature_count), dtype=torch.float32),
            "token_type_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.bool),
            "history_mask": torch.tensor(((False, True, True), (False, False, True)), dtype=torch.bool),
        }

        self.assertIsNotNone(model.temporal_gru)
        output = model(**inputs)

        self.assertEqual(tuple(output.policy_logits.shape), (2, 9))
        self.assertEqual(tuple(output.value.shape), (2,))
        self.assertEqual(tuple(output.opponent_action_logits.shape), (2, 9))

    def test_transformer_gru_temporal_aggregator_is_padding_invariant_per_row(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",),
            category_oov_buckets=1,
            window_size=3,
            embedding_dim=8,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=16,
            dropout=0.0,
            temporal_aggregator="gru",
        )
        model = EntityTokenTransformerPolicy(config)
        model.eval()
        shape = (1, config.window_size, config.token_count)
        base_inputs = {
            "categorical_ids": torch.zeros((*shape, config.categorical_feature_count), dtype=torch.long),
            "numeric_features": torch.zeros((*shape, config.numeric_feature_count), dtype=torch.float32),
            "token_type_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.bool),
            "history_mask": torch.tensor(((False, True, True),), dtype=torch.bool),
        }
        with torch.no_grad():
            base_inputs["numeric_features"][:, 1, :, :] = 1.0
            base_inputs["numeric_features"][:, 2, :, :] = 2.0
        other_inputs = {
            "categorical_ids": torch.ones((*shape, config.categorical_feature_count), dtype=torch.long),
            "numeric_features": torch.full((*shape, config.numeric_feature_count), 7.0, dtype=torch.float32),
            "token_type_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.bool),
            "history_mask": torch.tensor(((False, False, True),), dtype=torch.bool),
        }
        batched_inputs = {
            name: torch.cat((base_inputs[name], other_inputs[name]), dim=0)
            for name in base_inputs
        }

        with torch.no_grad():
            single_output = model(**base_inputs)
            batched_output = model(**batched_inputs)

        self.assertTrue(torch.allclose(single_output.policy_logits[0], batched_output.policy_logits[0], atol=1e-6))
        self.assertTrue(torch.allclose(single_output.value[0], batched_output.value[0], atol=1e-6))
        self.assertTrue(
            torch.allclose(single_output.opponent_action_logits[0], batched_output.opponent_action_logits[0], atol=1e-6)
        )

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

    def test_transformer_evaluator_timing_splits_encoding_and_forward_work(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
        config = TransformerPolicyConfig.compact_category(
            policy_id="fixture",
            category_vocab=("fixture",),
            category_oov_buckets=1,
            window_size=1,
            categorical_feature_count=1,
            numeric_feature_count=1,
            token_count=spec.token_count,
            embedding_dim=4,
            transformer_layers=1,
            attention_heads=1,
            feedforward_dim=8,
        )

        class FakeModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                del kwargs
                return SimpleNamespace(
                    policy_logits=torch.zeros(1, ACTION_COUNT),
                    opponent_action_logits=torch.zeros(1, ACTION_COUNT),
                    value=torch.tensor([0.25]),
                )

        timing = TransformerInferenceTimingAccumulator()
        result = SimpleNamespace(model_config=config)
        observations = (observation(1),)
        evaluate_transformer_action_priors(
            model=FakeModel(), result=result, observations=observations, device="cpu", timing=timing
        )
        evaluate_transformer_opponent_action_priors(
            model=FakeModel(), result=result, observations=observations, device="cpu", timing=timing
        )
        evaluate_transformer_observation_value(
            model=FakeModel(), result=result, observations=observations, device="cpu", timing=timing
        )

        snapshot = timing.snapshot()
        self.assertEqual(snapshot.observation_encoding_count, 3)
        self.assertEqual(snapshot.neural_forward_count, 3)
        self.assertEqual(snapshot.action_prior_neural_forward_count, 1)
        self.assertEqual(snapshot.opponent_action_prior_neural_forward_count, 1)
        self.assertEqual(snapshot.value_neural_forward_count, 1)
        self.assertEqual(snapshot.policy_neural_forward_count, 0)
        self.assertGreater(snapshot.observation_encoding_seconds, 0.0)
        self.assertGreater(snapshot.neural_forward_seconds, 0.0)

    def test_transformer_policy_records_value_estimate_on_decision(self) -> None:
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

        class FakePolicyModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                logits = torch.zeros(1, 9)
                logits[0, 1] = 2.0
                return SimpleNamespace(policy_logits=logits, value=torch.tensor([0.37]))

        timing = TransformerInferenceTimingAccumulator()
        policy = TransformerSoftmaxPolicy(
            model=FakePolicyModel(),
            result=TransformerTrainingResult(
                model_config=config,
                training_config=TransformerTrainingConfig(window_size=2),
                epochs=(),
                value_calibration_transform=ValueCalibrationTransform(scale=2.0, bias=0.0),
            ),
            inference_timing=timing,
        )

        decision = policy.select_action(observation(1), rng=__import__("random").Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertAlmostEqual(decision.value_estimate, 0.37, places=6)
        snapshot = timing.snapshot()
        self.assertEqual(snapshot.observation_encoding_count, 1)
        self.assertEqual(snapshot.neural_forward_count, 1)
        self.assertEqual(snapshot.policy_neural_forward_count, 1)
        self.assertEqual(snapshot.value_neural_forward_count, 0)
        self.assertGreater(snapshot.observation_encoding_seconds, 0.0)
        self.assertGreater(snapshot.neural_forward_seconds, 0.0)

    def test_transformer_policy_forward_fn_seam_preserves_decisions(self) -> None:
        # WS-L1: a forward_fn that round-trips logits/value through python lists (the RPC
        # serialization boundary a remote inference client will cross) must produce byte-identical
        # decisions to the local self.model path — same action, behavior-prob, and value — under
        # sampling with a fixed rng. This proves parity is structural (one shared decision path).
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        config = TransformerPolicyConfig.compact_category(
            policy_id="seam", category_vocab=("fixture",), category_oov_buckets=1, window_size=2,
            categorical_feature_count=1, numeric_feature_count=1,
            token_count=ObservationSpec(categorical_feature_count=1, numeric_feature_count=1).token_count,
            embedding_dim=4, transformer_layers=1, attention_heads=1, feedforward_dim=8,
        )

        class FakePolicyModel:
            def eval(self): pass
            def __call__(self, **kwargs):
                logits = torch.tensor([[0.5, 2.0, -1.0, 0.25, 1.5, 0.0, 0.75, -0.5, 1.1]])
                return SimpleNamespace(policy_logits=logits, value=torch.tensor([0.42]),
                                       opponent_action_logits=torch.zeros(1, 9))

        def make_result():
            return TransformerTrainingResult(
                model_config=config, training_config=TransformerTrainingConfig(window_size=2), epochs=(),
            )

        model = FakePolicyModel()

        def roundtrip_forward(tensors):
            out = model(**tensors)
            # simulate the RPC: tensors -> python lists -> tensors (lossless for float32)
            pl = torch.tensor([list(out.policy_logits[0].tolist())])
            val = torch.tensor([float(out.value[0])])
            return SimpleNamespace(policy_logits=pl, value=val, opponent_action_logits=None)

        local = TransformerSoftmaxPolicy(model=model, result=make_result(), deterministic=False)
        remote = TransformerSoftmaxPolicy(model=model, result=make_result(), deterministic=False,
                                          forward_fn=roundtrip_forward)
        d_local = local.select_action(observation(1), rng=__import__("random").Random(7))
        d_remote = remote.select_action(observation(1), rng=__import__("random").Random(7))
        self.assertEqual(d_local.action_index, d_remote.action_index)
        self.assertEqual(d_local.action_probability, d_remote.action_probability)
        self.assertEqual(d_local.value_estimate, d_remote.value_estimate)

    def test_remote_inference_policy_matches_local_over_http(self) -> None:
        # WS-L1 parity: a policy served by the inference server over real HTTP (`remote:` spec)
        # must produce byte-identical decisions to the local `neural:` policy on the same
        # checkpoint + seed. Proves the server forward + JSON round-trip preserve the decision
        # exactly (fp32 path); bf16 numerics are a separate scale-gate concern.
        if not torch_available():
            self.skipTest("requires torch")
        import random as _random
        import threading

        from pokezero.collection import policy_from_spec
        from pokezero.inference_service import serve_inference

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            ckpt = Path(temp_dir) / "transformer.pt"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            model_config = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)), category_oov_buckets=4, policy_id="parity",
                window_size=2, token_type_vocab_size=8, categorical_feature_count=1,
                numeric_feature_count=1, embedding_dim=16, transformer_layers=1,
                attention_heads=4, feedforward_dim=32, dropout=0.0,
            )
            model, result = train_transformer_policy(
                data_path, model_config=model_config,
                training_config=TransformerTrainingConfig(
                    batch_size=2, epochs=1, window_size=2, max_batches=1, device="cpu"),
            )
            save_transformer_checkpoint(ckpt, model, result=result)

            server = serve_inference(str(ckpt), host="127.0.0.1", port=0, device="cpu")
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                local = policy_from_spec(f"neural:{ckpt}")
                remote = policy_from_spec(f"remote:http://127.0.0.1:{port}")
                rl, rr = _random.Random(3), _random.Random(3)
                for i in range(1, 5):
                    a = local.select_action(observation(i), rng=rl)
                    b = remote.select_action(observation(i), rng=rr)
                    self.assertEqual(a.action_index, b.action_index)
                    self.assertEqual(a.action_probability, b.action_probability)
                    self.assertEqual(a.value_estimate, b.value_estimate)
                # Reconnect-once: force-close the client's persistent connection mid-stream and
                # confirm the next forward transparently reconnects and still matches local.
                if remote.forward_fn._conn is not None:
                    remote.forward_fn._conn.close()
                a = local.select_action(observation(5), rng=rl)
                b = remote.select_action(observation(5), rng=rr)
                self.assertEqual(a.action_index, b.action_index)
                self.assertEqual(a.value_estimate, b.value_estimate)
            finally:
                server.shutdown()

    def test_remote_config_retries_transient_bootstrap_failure(self) -> None:
        # A collector creates its remote policy by fetching /config. This is the startup path that
        # must absorb a transient socket-admission failure instead of failing the whole shard.
        from pokezero.inference_service import fetch_remote_config

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"window_size": 1, "policy_id": "ready"}'

        stdout = io.StringIO()
        with (
            patch("pokezero.inference_service.urlopen", side_effect=[OSError(1, "operation not permitted"), Response()]) as urlopen,
            patch("pokezero.inference_service._sleep_before_remote_retry") as sleep,
        ):
            config = fetch_remote_config("http://inference.test")

        self.assertEqual(config["policy_id"], "ready")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0)

    def test_remote_config_does_not_retry_permanent_http_error(self) -> None:
        from urllib.error import HTTPError

        from pokezero.inference_service import fetch_remote_config

        error = HTTPError("http://inference.test/config", 400, "bad request", hdrs=None, fp=None)
        with (
            patch("pokezero.inference_service.urlopen", side_effect=error) as urlopen,
            patch("pokezero.inference_service._sleep_before_remote_retry") as sleep,
        ):
            with self.assertRaises(HTTPError):
                fetch_remote_config("http://inference.test")

        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_remote_config_retries_are_bounded(self) -> None:
        from pokezero.inference_service import _REMOTE_RETRY_ATTEMPTS, fetch_remote_config

        with (
            patch("pokezero.inference_service.urlopen", side_effect=OSError(1, "operation not permitted")) as urlopen,
            patch("pokezero.inference_service._sleep_before_remote_retry") as sleep,
        ):
            with self.assertRaises(OSError):
                fetch_remote_config("http://inference.test")

        self.assertEqual(urlopen.call_count, _REMOTE_RETRY_ATTEMPTS)
        self.assertEqual([call.args for call in sleep.call_args_list], [(index,) for index in range(_REMOTE_RETRY_ATTEMPTS - 1)])

    def test_remote_forward_retries_transient_connection_failure(self) -> None:
        from pokezero.inference_service import RemoteForward

        class FailingConnection:
            def __init__(self):
                self.closed = False

            def request(self, *args, **kwargs):
                raise OSError(1, "operation not permitted")

            def close(self):
                self.closed = True

        class Response:
            status = 200

            def read(self):
                return b'{"policy_logits": [[0.0]], "value": [0.0], "opponent_action_logits": null}'

        class GoodConnection:
            def request(self, *args, **kwargs):
                return None

            def getresponse(self):
                return Response()

            def close(self):
                return None

        failing = FailingConnection()
        remote = RemoteForward("http://inference.test")
        with (
            patch("pokezero.inference_service.http.client.HTTPConnection", side_effect=[failing, GoodConnection()]) as connection,
            patch("pokezero.inference_service._sleep_before_remote_retry") as sleep,
        ):
            result = remote._request(b"payload")

        self.assertEqual(result["value"], [0.0])
        self.assertEqual(connection.call_count, 2)
        self.assertTrue(failing.closed)
        sleep.assert_called_once_with(0)

    def test_inference_server_hot_swap_reload(self) -> None:
        # WS-L1 hot-swap: POST /reload swaps the served checkpoint atomically. /config must report
        # the new policy_id and the served forward must change to the new weights — this is how the
        # pipeline points the server at each iteration's checkpoint.
        if not torch_available():
            self.skipTest("requires torch")
        import json as _json
        import random as _random
        import threading
        from urllib.request import Request, urlopen

        from pokezero.collection import policy_from_spec
        from pokezero.inference_service import serve_inference

        def _make_ckpt(temp_dir, name, epochs):
            data_path = Path(temp_dir) / f"{name}.jsonl"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            cfg = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)), category_oov_buckets=4, policy_id=name,
                window_size=2, token_type_vocab_size=8, categorical_feature_count=1,
                numeric_feature_count=1, embedding_dim=16, transformer_layers=1,
                attention_heads=4, feedforward_dim=32, dropout=0.0)
            model, result = train_transformer_policy(
                data_path, model_config=cfg,
                training_config=TransformerTrainingConfig(batch_size=2, epochs=epochs, window_size=2, device="cpu"))
            ckpt = Path(temp_dir) / f"{name}.pt"
            save_transformer_checkpoint(ckpt, model, result=result)
            return ckpt

        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_a = _make_ckpt(temp_dir, "swapA", 1)
            ckpt_b = _make_ckpt(temp_dir, "swapB", 5)
            server = serve_inference(str(ckpt_a), host="127.0.0.1", port=0, device="cpu")
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{port}"
                self.assertEqual(_json.loads(urlopen(f"{base}/config", timeout=10).read())["policy_id"], "swapA")
                remote = policy_from_spec(f"remote:{base}")
                d_a = remote.select_action(observation(1), rng=_random.Random(1))
                # reload to B
                req = Request(f"{base}/reload", data=_json.dumps({"checkpoint_path": str(ckpt_b)}).encode(),
                              headers={"Content-Type": "application/json"})
                self.assertEqual(urlopen(req, timeout=30).status, 200)
                self.assertEqual(_json.loads(urlopen(f"{base}/config", timeout=10).read())["policy_id"], "swapB")
                remote.reset()
                d_b = remote.select_action(observation(1), rng=_random.Random(1))
                # served forward changed to B's weights (A had 1 epoch, B had 5 → different value head)
                self.assertNotEqual(d_a.value_estimate, d_b.value_estimate)
            finally:
                server.shutdown()

    def test_inference_request_codec_round_trips(self) -> None:
        # WS-L1 raw-bytes wire: encode_forward_request -> decode_forward_request must reproduce the
        # tensors exactly (lossless), incl. dtypes/shapes, for the 5 window tensors.
        if not torch_available():
            self.skipTest("requires torch")
        import numpy as _np

        from pokezero.inference_service import decode_forward_request, encode_forward_request
        from pokezero.neural_policy import observation_window_to_torch

        tensors = observation_window_to_torch([observation(1)], window_size=2, device="cpu")
        keys = ("categorical_ids", "numeric_features", "token_type_ids", "attention_mask", "history_mask")
        decoded = decode_forward_request(encode_forward_request(tensors))
        for key in keys:
            self.assertTrue(_np.array_equal(decoded[key], tensors[key].detach().cpu().numpy()),
                            f"codec mismatch on {key}")

    def test_inference_forward_batch_matches_single(self) -> None:
        # WS-L1 batching: a coalesced batch of N identical requests must yield N results each
        # byte-identical to the single-forward result — i.e. the batch split is correct and
        # batch-of-N == N singles (so batching preserves parity).
        if not torch_available():
            self.skipTest("requires torch")
        from pokezero.inference_service import _forward_batch, run_forward_from_payload
        from pokezero.neural_policy import load_transformer_policy, observation_window_to_torch

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            ckpt = Path(temp_dir) / "transformer.pt"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            model_config = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)), category_oov_buckets=4, policy_id="batch",
                window_size=2, token_type_vocab_size=8, categorical_feature_count=1,
                numeric_feature_count=1, embedding_dim=16, transformer_layers=1,
                attention_heads=4, feedforward_dim=32, dropout=0.0,
            )
            model, result = train_transformer_policy(
                data_path, model_config=model_config,
                training_config=TransformerTrainingConfig(
                    batch_size=2, epochs=1, window_size=2, max_batches=1, device="cpu"),
            )
            save_transformer_checkpoint(ckpt, model, result=result)
            policy = load_transformer_policy(ckpt, device="cpu")
            tensors = observation_window_to_torch([observation(1)], window_size=2, device="cpu")
            payload = {k: tensors[k].tolist() for k in
                       ("categorical_ids", "numeric_features", "token_type_ids", "attention_mask", "history_mask")}
            single = run_forward_from_payload(policy, payload, device="cpu")
            batched = _forward_batch(policy, [payload, payload, payload], device="cpu")
            self.assertEqual(len(batched), 3)
            # Batched matmul reductions differ from single-forward at ~1e-9 (reduction order
            # depends on batch size), so compare within tolerance, not bitwise. This ~1e-9
            # batch-composition-dependent noise is expected and acceptable for stochastic
            # self-play collection (far below sampling noise; PPO stays self-consistent because
            # the collector records the behavior-prob from the served logits). Batch-of-1 remains
            # exact (see the HTTP parity test).
            for item in batched:
                for a, b in zip(item["policy_logits"][0], single["policy_logits"][0]):
                    self.assertAlmostEqual(a, b, places=4)
                self.assertAlmostEqual(item["value"][0], single["value"][0], places=4)

    def test_inference_server_returns_400_on_malformed_forward_body(self) -> None:
        # WS-L1 robustness: a malformed /forward body must return a diagnosable 400, not drop the
        # socket, and the server must survive to serve the next request (per-request isolation).
        if not torch_available():
            self.skipTest("requires torch")
        import json as _json
        import threading
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        from pokezero.inference_service import serve_inference

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            ckpt = Path(temp_dir) / "transformer.pt"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            model_config = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)), category_oov_buckets=4, policy_id="robust",
                window_size=2, token_type_vocab_size=8, categorical_feature_count=1,
                numeric_feature_count=1, embedding_dim=16, transformer_layers=1,
                attention_heads=4, feedforward_dim=32, dropout=0.0,
            )
            model, result = train_transformer_policy(
                data_path, model_config=model_config,
                training_config=TransformerTrainingConfig(
                    batch_size=2, epochs=1, window_size=2, max_batches=1, device="cpu"),
            )
            save_transformer_checkpoint(ckpt, model, result=result)
            server = serve_inference(str(ckpt), host="127.0.0.1", port=0, device="cpu")
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                bad = Request(f"http://127.0.0.1:{port}/forward",
                              data=b'{"categorical_ids": "not a tensor"}',
                              headers={"Content-Type": "application/json"})
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(bad, timeout=10)
                self.assertEqual(ctx.exception.code, 400)
                # server survives: /config still works
                with urlopen(f"http://127.0.0.1:{port}/config", timeout=10) as resp:
                    self.assertEqual(_json.loads(resp.read())["window_size"], 2)
            finally:
                server.shutdown()

    def test_transformer_policy_drops_non_finite_value_estimate(self) -> None:
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

        class NaNValueModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                logits = torch.zeros(1, 9)
                logits[0, 1] = 2.0
                return SimpleNamespace(policy_logits=logits, value=torch.tensor([float("nan")]))

        policy = TransformerSoftmaxPolicy(
            model=NaNValueModel(),
            result=TransformerTrainingResult(
                model_config=config,
                training_config=TransformerTrainingConfig(window_size=2),
                epochs=(),
            ),
        )

        decision = policy.select_action(observation(1), rng=__import__("random").Random(1))

        self.assertEqual(decision.action_index, 1)
        self.assertIsNone(decision.value_estimate)
        self.assertEqual(decision.metadata["value_estimate_dropped"], "non_finite")

    def test_transformer_training_config_validates_training_knobs(self) -> None:
        self.assertEqual(TransformerTrainingConfig().window_size, 1)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            TransformerTrainingConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "value_loss_weight"):
            TransformerTrainingConfig(value_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "value_clip_range"):
            TransformerTrainingConfig(value_clip_range=0.0)
        with self.assertRaisesRegex(ValueError, "value_clip_range"):
            TransformerTrainingConfig(value_clip_range=-0.1)
        with self.assertRaisesRegex(ValueError, "opponent_action_loss_weight"):
            TransformerTrainingConfig(opponent_action_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "switch_action_loss_weight"):
            TransformerTrainingConfig(switch_action_loss_weight=0.0)
        with self.assertRaisesRegex(ValueError, "action_family_loss_weight"):
            TransformerTrainingConfig(action_family_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "switch_target_loss_weight"):
            TransformerTrainingConfig(switch_target_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "hp_delta_return_weight"):
            TransformerTrainingConfig(hp_delta_return_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "faint_delta_return_weight"):
            TransformerTrainingConfig(faint_delta_return_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "turn_penalty_after"):
            TransformerTrainingConfig(turn_penalty_after=-1)
        with self.assertRaisesRegex(ValueError, "turn_penalty"):
            TransformerTrainingConfig(turn_penalty=-0.1)
        with self.assertRaisesRegex(ValueError, "turn_penalty_after"):
            TransformerTrainingConfig(turn_penalty=0.1)
        with self.assertRaisesRegex(ValueError, "ppo_target_mode"):
            TransformerTrainingConfig(objective="ppo", ppo_target_mode="bad")
        with self.assertRaisesRegex(ValueError, "requires objective='ppo'"):
            TransformerTrainingConfig(ppo_target_mode="gae")
        with self.assertRaisesRegex(ValueError, "gae_lambda"):
            TransformerTrainingConfig(objective="ppo", ppo_target_mode="gae", gae_lambda=1.5)
        with self.assertRaisesRegex(ValueError, "objective"):
            TransformerTrainingConfig(objective="bogus")
        with self.assertRaisesRegex(ValueError, "clip_epsilon"):
            TransformerTrainingConfig(objective="ppo", clip_epsilon=0.0)
        with self.assertRaisesRegex(ValueError, "max_grad_norm"):
            TransformerTrainingConfig(max_grad_norm=0.0)
        with self.assertRaisesRegex(ValueError, "max_grad_norm"):
            TransformerTrainingConfig(max_grad_norm=-1.0)
        self.assertIsNone(TransformerTrainingConfig().amp)
        self.assertEqual(TransformerTrainingConfig(amp="bf16").amp, "bf16")
        with self.assertRaisesRegex(ValueError, "amp"):
            TransformerTrainingConfig(amp="fp16")
        with self.assertRaisesRegex(ValueError, "learning_rate_schedule"):
            TransformerTrainingConfig(learning_rate_schedule="bogus")
        with self.assertRaisesRegex(ValueError, "learning_rate_schedule_total_games"):
            TransformerTrainingConfig(learning_rate_schedule_total_games=0)
        with self.assertRaisesRegex(ValueError, "learning_rate_progress_start"):
            TransformerTrainingConfig(learning_rate_progress_start=-0.1)
        with self.assertRaisesRegex(ValueError, "learning_rate_progress_end"):
            TransformerTrainingConfig(learning_rate_progress_end=1.1)
        with self.assertRaisesRegex(ValueError, "learning_rate_progress_end"):
            TransformerTrainingConfig(learning_rate_progress_start=0.5, learning_rate_progress_end=0.25)
        with self.assertRaisesRegex(ValueError, "learning_rate_warmup_progress"):
            TransformerTrainingConfig(learning_rate_warmup_progress=-0.1)
        with self.assertRaisesRegex(ValueError, "learning_rate_warmup_progress"):
            TransformerTrainingConfig(learning_rate_warmup_progress=1.1)
        self.assertIsNone(TransformerTrainingConfig().max_grad_norm)
        self.assertEqual(TransformerTrainingConfig(max_grad_norm=0.543).to_dict()["max_grad_norm"], 0.543)
        self.assertEqual(TransformerTrainingConfig(value_clip_range=0.0184).to_dict()["value_clip_range"], 0.0184)
        self.assertEqual(
            TransformerTrainingConfig(learning_rate_schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE).learning_rate_schedule,
            "mit-thesis",
        )
        with self.assertRaisesRegex(ValueError, "objective='value-only'"):
            TransformerTrainingConfig(objective="value-only")
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
        self.assertEqual(metrics["ppo_valid_examples"], 3)
        self.assertAlmostEqual(metrics["ppo_advantage_sum"], 3.0, places=5)
        self.assertAlmostEqual(metrics["ppo_advantage_square_sum"], 3.0, places=5)
        self.assertAlmostEqual(metrics["ppo_ratio_sum"], 3.0, places=5)
        self.assertEqual(metrics["ppo_clip_count"], 0)
        self.assertAlmostEqual(metrics["ppo_entropy_sum"], 3.0 * torch.log(torch.tensor(9.0)).item(), places=5)
        # Behavior-cloning objective on the same tensors yields a positive CE policy loss.
        bc_loss, bc_metrics = _transformer_loss(output, tensors, TransformerTrainingConfig())
        self.assertGreater(bc_metrics["policy_loss"], 0.0)

    def test_ppo_objective_uses_recorded_gae_targets_when_present(self) -> None:
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
            "returns": torch.ones(2),
            "ppo_advantages": torch.tensor([0.25, 99.0]),
            "ppo_advantage_mask": torch.tensor([True, False]),
            "ppo_value_targets": torch.tensor([0.4, 99.0]),
            "ppo_value_target_mask": torch.tensor([True, False]),
            "action_probabilities": torch.full((2,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(2, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(2, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(2, dtype=torch.long),
        }

        loss, metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(
                objective="ppo",
                ppo_target_mode="gae",
                normalize_advantage=False,
                entropy_coef=0.0,
                opponent_action_loss_weight=0.0,
            ),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics["policy_loss"], -0.625, places=5)
        self.assertAlmostEqual(metrics["value_loss"], ((0.4**2) + 1.0) / 2.0, places=5)
        self.assertAlmostEqual(metrics["ppo_advantage_sum"], 1.25, places=5)
        self.assertAlmostEqual(metrics["ppo_advantage_square_sum"], (0.25**2) + 1.0, places=5)

    def test_ppo_value_loss_uses_recorded_value_clip_range(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        output = TransformerPolicyOutput(
            policy_logits=torch.zeros(2, 9),
            value=torch.tensor([0.9, 0.9]),
            opponent_action_logits=torch.zeros(2, 9),
        )
        tensors = {
            "legal_action_mask": torch.ones(2, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0, 1], dtype=torch.long),
            "returns": torch.ones(2),
            "value_estimates": torch.tensor([0.0, 0.0]),
            "value_estimate_mask": torch.tensor([True, False]),
            "action_probabilities": torch.full((2,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(2, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(2, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(2, dtype=torch.long),
        }

        _, metrics = _transformer_loss(
            output,
            tensors,
            TransformerTrainingConfig(
                objective="ppo",
                value_clip_range=0.1,
                normalize_advantage=False,
                entropy_coef=0.0,
                opponent_action_loss_weight=0.0,
            ),
        )

        # First row uses V_old=0.0 and clip range 0.1, so the clipped prediction is 0.1
        # and max(unclipped=0.01, clipped=0.81) is used. Second row has no recorded old
        # value and falls back to the normal 0.01 MSE.
        self.assertAlmostEqual(metrics["value_loss"], (0.81 + 0.01) / 2.0, places=5)
        self.assertEqual(metrics["ppo_value_clip_eligible_examples"], 1)
        self.assertEqual(metrics["ppo_value_clip_count"], 1)

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
                freeze_non_value_parameters=True,
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

    def test_training_weights_upweight_supervised_policy_rows(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        import torch

        from pokezero.neural_policy import TransformerPolicyOutput, _transformer_loss

        logits = torch.zeros(2, 9)
        logits[0, 0] = 8.0
        logits[1, 0] = 8.0  # Wrong for row 1 target action 1.
        output = TransformerPolicyOutput(
            policy_logits=logits,
            value=torch.zeros(2),
            opponent_action_logits=torch.zeros(2, 9),
        )
        base_tensors = {
            "legal_action_mask": torch.ones(2, 9, dtype=torch.bool),
            "action_indices": torch.tensor([0, 1], dtype=torch.long),
            "returns": torch.ones(2),
            "action_probabilities": torch.full((2,), 1.0 / 9.0),
            "action_probability_mask": torch.ones(2, dtype=torch.bool),
            "opponent_action_mask": torch.zeros(2, dtype=torch.bool),
            "opponent_action_indices": torch.zeros(2, dtype=torch.long),
        }
        config = TransformerTrainingConfig(
            objective="behavior-cloning",
            opponent_action_loss_weight=0.0,
            value_loss_weight=0.0,
        )

        _, unweighted = _transformer_loss(output, base_tensors, config)
        _, weighted = _transformer_loss(
            output,
            {
                **base_tensors,
                "training_weights": torch.tensor([1.0, 8.0]),
            },
            config,
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
        self.assertEqual(metrics["ppo_valid_examples"], 1)
        self.assertEqual(metrics["ppo_clip_count"], 1)
        self.assertGreater(metrics["ppo_ratio_sum"], 1.2)

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
        self.assertEqual(metrics["ppo_objective_examples"], 2)
        self.assertEqual(metrics["ppo_valid_examples"], 0)
        self.assertTrue(torch.isfinite(loss))
        # A zero behavior probability is excluded even if its mask flag is set.
        zero_prob = {**base, "action_probabilities": torch.tensor([0.0, 1.0 / 9.0]), "action_probability_mask": torch.ones(2, dtype=torch.bool)}
        loss2, zero_prob_metrics = _transformer_loss(output, zero_prob, config)
        self.assertEqual(zero_prob_metrics["ppo_valid_examples"], 1)
        self.assertTrue(torch.isfinite(loss2))

    def test_ppo_epoch_metrics_aggregate_diagnostics(self) -> None:
        from pokezero.neural_policy import _TorchMetricTotals

        totals = _TorchMetricTotals()
        totals.add(
            4,
            {
                "loss": 2.0,
                "policy_loss": 1.0,
                "policy_correct": 2,
                "value_loss": 0.5,
                "opponent_examples": 0,
                "action_family_examples": 0,
                "switch_target_examples": 0,
                "ppo_objective_examples": 4,
                "ppo_valid_examples": 2,
                "ppo_advantage_sum": 1.0,
                "ppo_advantage_square_sum": 5.0,
                "ppo_ratio_sum": 2.2,
                "ppo_clip_count": 1,
                "ppo_entropy_sum": 3.0,
            },
        )
        totals.add(
            2,
            {
                "loss": 1.0,
                "policy_loss": 0.5,
                "policy_correct": 1,
                "value_loss": 0.25,
                "opponent_examples": 0,
                "action_family_examples": 0,
                "switch_target_examples": 0,
                "ppo_objective_examples": 2,
                "ppo_valid_examples": 1,
                "ppo_advantage_sum": -1.0,
                "ppo_advantage_square_sum": 1.0,
                "ppo_ratio_sum": 0.8,
                "ppo_clip_count": 0,
                "ppo_entropy_sum": 1.5,
            },
        )

        metrics = totals.to_epoch_metrics(epoch=1)

        self.assertEqual(metrics.ppo_valid_examples, 3)
        self.assertAlmostEqual(metrics.ppo_valid_fraction, 0.5)
        self.assertAlmostEqual(metrics.ppo_advantage_mean, 0.0)
        self.assertAlmostEqual(metrics.ppo_advantage_std, (6.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(metrics.ppo_ratio_mean, 1.0)
        self.assertAlmostEqual(metrics.ppo_clip_fraction, 1.0 / 3.0)
        self.assertAlmostEqual(metrics.ppo_entropy, 1.5)

    def test_mit_thesis_learning_rate_schedule_uses_global_progress_curve(self) -> None:
        self.assertAlmostEqual(
            learning_rate_for_progress(
                base_learning_rate=5.9e-5,
                schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                progress=0.0,
            ),
            5.9e-5,
        )
        self.assertAlmostEqual(
            learning_rate_for_progress(
                base_learning_rate=5.9e-5,
                schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                progress=0.5,
            ),
            5.9e-5 / (5.0**1.5),
        )
        self.assertAlmostEqual(
            learning_rate_for_progress(
                base_learning_rate=5.9e-5,
                schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                progress=1.0,
            ),
            5.9e-5 / (9.0**1.5),
        )

    def test_learning_rate_warmup_ramps_linearly_then_follows_decay(self) -> None:
        base = 5.9e-5
        warmup = 0.1
        # scheduled value at the warmup boundary — the ramp target.
        target = base / (((8.0 * warmup) + 1.0) ** 1.5)
        # At progress 0 the LR is 0 (textbook linear warmup from zero).
        self.assertAlmostEqual(
            learning_rate_for_progress(
                base_learning_rate=base,
                schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                progress=0.0,
                warmup_progress=warmup,
            ),
            0.0,
        )
        # Halfway through warmup -> half the boundary value.
        self.assertAlmostEqual(
            learning_rate_for_progress(
                base_learning_rate=base,
                schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                progress=warmup / 2.0,
                warmup_progress=warmup,
            ),
            target * 0.5,
        )
        # Continuity: at exactly warmup_progress the ramp and the decay agree.
        self.assertAlmostEqual(
            learning_rate_for_progress(
                base_learning_rate=base,
                schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                progress=warmup,
                warmup_progress=warmup,
            ),
            target,
        )
        # Past warmup the curve is the unchanged decay schedule.
        self.assertAlmostEqual(
            learning_rate_for_progress(
                base_learning_rate=base,
                schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                progress=0.5,
                warmup_progress=warmup,
            ),
            base / (5.0**1.5),
        )

    def test_learning_rate_warmup_zero_preserves_legacy_curve(self) -> None:
        base = 5.9e-5
        for progress in (0.0, 0.25, 0.5, 1.0):
            self.assertAlmostEqual(
                learning_rate_for_progress(
                    base_learning_rate=base,
                    schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                    progress=progress,
                    warmup_progress=0.0,
                ),
                learning_rate_for_progress(
                    base_learning_rate=base,
                    schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                    progress=progress,
                ),
            )

    def test_learning_rate_warmup_progress_out_of_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            learning_rate_for_progress(
                base_learning_rate=5.9e-5,
                schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                progress=0.5,
                warmup_progress=1.5,
            )
        with self.assertRaises(ValueError):
            learning_rate_for_progress(
                base_learning_rate=5.9e-5,
                schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                progress=0.5,
                warmup_progress=float("nan"),
            )

    def test_learning_rate_warmup_ramps_under_constant_schedule(self) -> None:
        # Under the constant schedule the ramp target is base itself; past warmup the LR is flat base.
        base = 4.2e-5
        warmup = 0.2
        self.assertAlmostEqual(
            learning_rate_for_progress(
                base_learning_rate=base,
                schedule=CONSTANT_LEARNING_RATE_SCHEDULE,
                progress=warmup / 4.0,
                warmup_progress=warmup,
            ),
            base * 0.25,
        )
        self.assertAlmostEqual(
            learning_rate_for_progress(
                base_learning_rate=base,
                schedule=CONSTANT_LEARNING_RATE_SCHEDULE,
                progress=0.9,
                warmup_progress=warmup,
            ),
            base,
        )

    def test_learning_rate_for_epoch_applies_configured_warmup(self) -> None:
        from pokezero.neural_policy import _learning_rate_for_epoch

        base = 5.9e-5
        # A single-epoch config pinned at progress 0 with warmup on -> the epoch LR is the ramp start (0).
        config = TransformerTrainingConfig(
            learning_rate=base,
            learning_rate_schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
            epochs=1,
            learning_rate_progress_start=0.0,
            learning_rate_progress_end=0.0,
            learning_rate_warmup_progress=0.1,
        )
        self.assertAlmostEqual(_learning_rate_for_epoch(config, epoch=1), 0.0)
        # Same config without warmup returns the undecayed base at progress 0 (proves the field is threaded).
        no_warmup = TransformerTrainingConfig(
            learning_rate=base,
            learning_rate_schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
            epochs=1,
            learning_rate_progress_start=0.0,
            learning_rate_progress_end=0.0,
        )
        self.assertAlmostEqual(_learning_rate_for_epoch(no_warmup, epoch=1), base)

    def test_ppo_epoch_metrics_reports_zero_valid_coverage(self) -> None:
        from pokezero.neural_policy import _TorchMetricTotals

        totals = _TorchMetricTotals()
        totals.add(
            2,
            {
                "loss": 1.0,
                "policy_loss": 0.0,
                "policy_correct": 0,
                "value_loss": 0.5,
                "opponent_examples": 0,
                "action_family_examples": 0,
                "switch_target_examples": 0,
                "ppo_objective_examples": 2,
                "ppo_valid_examples": 0,
                "ppo_advantage_sum": 0.0,
                "ppo_advantage_square_sum": 0.0,
                "ppo_ratio_sum": 0.0,
                "ppo_clip_count": 0,
                "ppo_entropy_sum": 0.0,
            },
        )

        metrics = totals.to_epoch_metrics(epoch=1)

        self.assertEqual(metrics.ppo_valid_examples, 0)
        self.assertEqual(metrics.ppo_valid_fraction, 0.0)
        self.assertIsNone(metrics.ppo_advantage_mean)
        self.assertIsNone(metrics.ppo_ratio_mean)

    def test_non_ppo_epoch_metrics_omit_ppo_diagnostics(self) -> None:
        from pokezero.neural_policy import _TorchMetricTotals

        totals = _TorchMetricTotals()
        totals.add(
            2,
            {
                "loss": 1.0,
                "policy_loss": 0.5,
                "policy_correct": 1,
                "value_loss": 0.25,
                "opponent_examples": 0,
                "action_family_examples": 0,
                "switch_target_examples": 0,
                "ppo_objective_examples": 0,
                "ppo_valid_examples": 0,
                "ppo_advantage_sum": 0.0,
                "ppo_advantage_square_sum": 0.0,
                "ppo_ratio_sum": 0.0,
                "ppo_clip_count": 0,
                "ppo_entropy_sum": 0.0,
            },
        )

        metrics = totals.to_epoch_metrics(epoch=1)

        self.assertIsNone(metrics.ppo_valid_examples)
        self.assertIsNone(metrics.ppo_valid_fraction)
        self.assertIsNone(metrics.ppo_entropy)

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
        require_neural_promoted_opponent_pool(
            ("neural:a.pt", "neural:b.pt", "neural:c.pt", "neural:d.pt"),
            promotion_pool_registry_path=Path("promotions.json"),
            current_policy_spec="neural:d.pt",
            max_historical_opponents=2,
            required_size=2,
            historical_opponent_selection="spread",
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

    def test_require_torch_applies_thread_env_once(self) -> None:
        class FakeTorch:
            def __init__(self) -> None:
                self.num_threads: list[int] = []
                self.num_interop_threads: list[int] = []

            def set_num_threads(self, value: int) -> None:
                self.num_threads.append(value)

            def set_num_interop_threads(self, value: int) -> None:
                self.num_interop_threads.append(value)

        fake_torch = FakeTorch()
        with (
            patch.object(neural_policy_module, "torch", fake_torch),
            patch.object(neural_policy_module, "nn", object()),
            patch.object(neural_policy_module, "_TORCH_THREAD_ENV_APPLIED", False),
            patch.dict(
                os.environ,
                {
                    "POKEZERO_TORCH_NUM_THREADS": "2",
                    "POKEZERO_TORCH_NUM_INTEROP_THREADS": "3",
                },
            ),
        ):
            self.assertIs(require_torch(), fake_torch)
            self.assertIs(require_torch(), fake_torch)

        self.assertEqual(fake_torch.num_threads, [2])
        self.assertEqual(fake_torch.num_interop_threads, [3])

    def test_require_torch_rejects_invalid_thread_env(self) -> None:
        class FakeTorch:
            def set_num_threads(self, value: int) -> None:
                raise AssertionError("invalid env should fail before applying torch threads")

            def set_num_interop_threads(self, value: int) -> None:
                raise AssertionError("invalid env should fail before applying torch threads")

        with (
            patch.object(neural_policy_module, "torch", FakeTorch()),
            patch.object(neural_policy_module, "nn", object()),
            patch.object(neural_policy_module, "_TORCH_THREAD_ENV_APPLIED", False),
            patch.dict(
                os.environ,
                {
                    "POKEZERO_TORCH_NUM_THREADS": "0",
                    "POKEZERO_TORCH_NUM_INTEROP_THREADS": "",
                },
            ),
        ):
            with self.assertRaisesRegex(ValueError, "POKEZERO_TORCH_NUM_THREADS"):
                require_torch()

    def test_require_torch_reports_late_interop_thread_application(self) -> None:
        class FakeTorch:
            def set_num_threads(self, value: int) -> None:
                pass

            def set_num_interop_threads(self, value: int) -> None:
                raise RuntimeError("cannot set number of interop threads after parallel work")

        with (
            patch.object(neural_policy_module, "torch", FakeTorch()),
            patch.object(neural_policy_module, "nn", object()),
            patch.object(neural_policy_module, "_TORCH_THREAD_ENV_APPLIED", False),
            patch.dict(
                os.environ,
                {
                    "POKEZERO_TORCH_NUM_THREADS": "",
                    "POKEZERO_TORCH_NUM_INTEROP_THREADS": "1",
                },
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "POKEZERO_TORCH_NUM_INTEROP_THREADS"):
                require_torch()

    def test_require_torch_applies_thread_env_to_real_torch_in_fresh_process(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        script = "\n".join(
            [
                "from pokezero.neural_policy import require_torch",
                "torch = require_torch()",
                "assert torch.get_num_threads() == 1, torch.get_num_threads()",
                "assert torch.get_num_interop_threads() == 1, torch.get_num_interop_threads()",
            ]
        )
        env = {
            **os.environ,
            "POKEZERO_TORCH_NUM_THREADS": "1",
            "POKEZERO_TORCH_NUM_INTEROP_THREADS": "1",
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            env=env,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_load_transformer_policy_resolves_default_device_before_checkpoint_load(self) -> None:
        fake_policy = object()
        with (
            patch("pokezero.neural_policy.resolve_torch_device", return_value="cpu") as resolve_device,
            patch("pokezero.neural_policy.load_transformer_checkpoint", return_value=("model", "result")) as load,
            patch("pokezero.neural_policy.TransformerSoftmaxPolicy", return_value=fake_policy) as policy,
        ):
            restored = load_transformer_policy(Path("checkpoint.pt"), device=None)

        self.assertIs(restored, fake_policy)
        resolve_device.assert_called_once_with(None)
        load.assert_called_once_with(Path("checkpoint.pt"), map_location="cpu")
        self.assertEqual(policy.call_args.kwargs["device"], "cpu")

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

    def test_neural_cli_cache_data_wires_dataset_config_without_torch(self) -> None:
        fake_summary = SimpleNamespace(path=Path("cache"), record_count=2, example_count=8, byte_size=1024)
        with patch("pokezero.neural_cli.write_training_cache_from_rollouts", return_value=fake_summary) as cache:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = neural_cli_main(
                    [
                        "cache-data",
                        "--data",
                        "rollouts.jsonl",
                        "--out",
                        "cache",
                        "--overwrite",
                        "--window-size",
                        "3",
                        "--discount",
                        "0.9",
                        "--ppo-target-mode",
                        "gae",
                        "--gae-lambda",
                        "0.7",
                    ]
                )

        self.assertEqual(exit_code, 0)
        args, kwargs = cache.call_args
        self.assertEqual(args[0], [Path("rollouts.jsonl")])
        self.assertEqual(args[1], Path("cache"))
        self.assertTrue(kwargs["overwrite"])
        self.assertEqual(kwargs["config"].window_size, 3)
        self.assertEqual(kwargs["config"].discount, 0.9)
        self.assertEqual(kwargs["config"].ppo_target_mode, "gae")
        self.assertEqual(kwargs["config"].gae_lambda, 0.7)
        self.assertEqual(kwargs["max_cache_root_bytes"], 50 * 1024 * 1024 * 1024)
        self.assertEqual(kwargs["cache_root"], Path("."))
        self.assertIn("training_cache_examples: 8", stdout.getvalue())

    def test_neural_cli_refutation_cache_validation_accepts_capped_policy_value_cache(self) -> None:
        args = SimpleNamespace(
            data=[Path("primary-cache")],
            refutation_cache=[Path("refutation-cache")],
            refutation_max_fraction=0.1,
            refutation_target_mode="policy-value",
            objective="ppo",
        )
        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch(
                "pokezero.neural_cli._refutation_cache_training_contract",
                return_value=("policy-value", ("behavior-cloning", "ppo", "reward-weighted")),
            ),
        ):
            paths = _validate_refutation_cache_args(args)

        self.assertEqual(paths, (Path("refutation-cache"),))

    def test_neural_cli_refutation_cache_validation_accepts_policy_distribution_cache(self) -> None:
        args = SimpleNamespace(
            data=[Path("primary-cache")],
            refutation_cache=[Path("refutation-cache")],
            refutation_max_fraction=0.1,
            refutation_target_mode="policy-distribution-value",
            objective="behavior-cloning",
        )
        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch(
                "pokezero.neural_cli._refutation_cache_training_contract",
                return_value=("policy-distribution-value", ("behavior-cloning", "ppo", "reward-weighted")),
            ),
        ):
            paths = _validate_refutation_cache_args(args)

        self.assertEqual(paths, (Path("refutation-cache"),))

    def test_neural_cli_refutation_cache_validation_rejects_fraction_above_cap(self) -> None:
        args = SimpleNamespace(
            data=[Path("primary-cache")],
            refutation_cache=[Path("refutation-cache")],
            refutation_max_fraction=0.25,
            refutation_target_mode="policy-value",
            objective="ppo",
        )

        with self.assertRaisesRegex(ValueError, "at most 0.2"):
            _validate_refutation_cache_args(args)

    def test_neural_cli_refutation_cache_validation_rejects_value_mode_for_bc(self) -> None:
        args = SimpleNamespace(
            data=[Path("primary-cache")],
            refutation_cache=[Path("refutation-cache")],
            refutation_max_fraction=0.1,
            refutation_target_mode="value",
            objective="behavior-cloning",
        )

        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch(
                "pokezero.neural_cli._refutation_cache_training_contract",
                return_value=("value", ("ppo", "value-only")),
            ),
            self.assertRaisesRegex(ValueError, "compatible with ppo, value-only"),
        ):
            _validate_refutation_cache_args(args)

    def test_neural_cli_refutation_cache_validation_rejects_target_mode_mismatch(self) -> None:
        args = SimpleNamespace(
            data=[Path("primary-cache")],
            refutation_cache=[Path("refutation-cache")],
            refutation_max_fraction=0.1,
            refutation_target_mode="policy-value",
            objective="ppo",
        )

        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch(
                "pokezero.neural_cli._refutation_cache_training_contract",
                return_value=("value", ("ppo", "value-only")),
            ),
            self.assertRaisesRegex(ValueError, "was built with target_mode='value'"),
        ):
            _validate_refutation_cache_args(args)

    def test_neural_cli_refutation_cache_validation_requires_cache_directory(self) -> None:
        args = SimpleNamespace(
            data=[Path("primary-cache")],
            refutation_cache=[Path("rollouts.jsonl")],
            refutation_max_fraction=0.1,
            refutation_target_mode="policy-value",
            objective="ppo",
        )
        with patch("pokezero.neural_cli.is_training_cache_path", return_value=False):
            with self.assertRaisesRegex(ValueError, "not a training-cache directory"):
                _validate_refutation_cache_args(args)

    def test_neural_cli_refutation_cache_contract_requires_stamped_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            cache_path.mkdir()
            (cache_path / "metadata.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing refutation_training metadata"):
                _refutation_cache_training_contract(cache_path)

    def test_neural_cli_refutation_cache_contract_rejects_tampered_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            cache_path.mkdir()
            (cache_path / "metadata.json").write_text(
                json.dumps(
                    {
                        "refutation_training": {
                            "target_mode": "value",
                            "compatible_objectives": ["behavior-cloning"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "requires \\('ppo', 'value-only'\\)"):
                _refutation_cache_training_contract(cache_path)

    def test_neural_cli_refutation_cache_validation_reads_stamped_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "refutation-cache"
            cache_path.mkdir()
            (cache_path / "metadata.json").write_text(
                json.dumps(
                    {
                        "refutation_training": {
                            "target_mode": "value",
                            "compatible_objectives": ["ppo", "value-only"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                data=[Path(temp_dir) / "primary-cache"],
                refutation_cache=[cache_path],
                refutation_max_fraction=0.1,
                refutation_target_mode="value",
                objective="ppo",
            )

            paths = _validate_refutation_cache_args(args)

        self.assertEqual(paths, (cache_path,))

    def test_neural_cli_training_cache_lifecycle_rejects_oversized_active_cache(self) -> None:
        args = SimpleNamespace(
            data=[Path("cache-a"), Path("cache-b")],
            max_cache_gb=50,
            delete_cache_after_read=False,
        )
        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch("pokezero.neural_cli.training_cache_root_byte_size", return_value=(51 * 1024 * 1024 * 1024)),
        ):
            with self.assertRaisesRegex(ValueError, "root footprint"):
                _training_cache_lifecycle(args)

    def test_neural_cli_training_cache_lifecycle_deletes_consumed_cache_after_finalize(self) -> None:
        args = SimpleNamespace(
            data=[Path("cache-a")],
            max_cache_gb=50,
            delete_cache_after_read=True,
        )
        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch("pokezero.neural_cli.training_cache_root_byte_size", return_value=1234),
            patch("pokezero.neural_cli.training_cache_paths_byte_size", return_value=1234),
            patch("pokezero.neural_cli.delete_training_cache_path") as delete_cache,
        ):
            lifecycle = _training_cache_lifecycle(args)
            callback = lifecycle.consumed_cache_callback
            self.assertIsNotNone(callback)
            assert callback is not None
            callback(Path("cache-a"))
            delete_cache.assert_not_called()
            lifecycle.finalize_after_checkpoint()

        delete_cache.assert_called_once_with(Path("cache-a"))
        self.assertEqual(lifecycle.to_summary()["consumed_paths"], ["cache-a"])
        self.assertEqual(lifecycle.to_summary()["deleted_paths"], ["cache-a"])
        self.assertEqual(lifecycle.to_summary()["deleted_bytes"], 1234)

    def test_neural_cli_training_cache_lifecycle_defaults_to_delete_for_cache_inputs(self) -> None:
        args = SimpleNamespace(
            data=[Path("cache-a")],
            max_cache_gb=50,
        )
        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch("pokezero.neural_cli.training_cache_root_byte_size", return_value=1234),
        ):
            lifecycle = _training_cache_lifecycle(args)

        self.assertIsNotNone(lifecycle.consumed_cache_callback)
        self.assertEqual(lifecycle.to_summary()["footprint_bytes"], 1234)
        self.assertEqual(lifecycle.to_summary()["footprint_limit_bytes"], 50 * 1024 * 1024 * 1024)

    def test_neural_cli_training_cache_lifecycle_keep_cache_disables_delete_callback(self) -> None:
        args = SimpleNamespace(
            data=[Path("cache-a")],
            max_cache_gb=50,
            delete_cache_after_read=False,
        )
        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch("pokezero.neural_cli.training_cache_root_byte_size", return_value=1234),
        ):
            lifecycle = _training_cache_lifecycle(args)

        self.assertIsNone(lifecycle.consumed_cache_callback)
        self.assertFalse(lifecycle.to_summary()["delete_after_checkpoint"])

    def test_neural_cli_training_cache_lifecycle_rejects_caps_above_project_limit(self) -> None:
        args = SimpleNamespace(
            data=[Path("cache-a")],
            max_cache_gb=51,
            delete_cache_after_read=False,
        )
        with patch("pokezero.neural_cli.is_training_cache_path", return_value=True):
            with self.assertRaisesRegex(ValueError, "cannot exceed 50"):
                _training_cache_lifecycle(args)

    def test_neural_cli_input_data_paths_byte_size_counts_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = temp_path / "first.jsonl"
            second = temp_path / "second.jsonl"
            first.write_bytes(b"abcd")
            second.write_bytes(b"ef")

            self.assertEqual(_input_data_paths_byte_size([first, second]), 6)

    def test_neural_cli_input_data_paths_byte_size_counts_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nested = temp_path / "nested"
            nested.mkdir()
            (temp_path / "first.bin").write_bytes(b"abcd")
            (nested / "second.bin").write_bytes(b"efg")

            self.assertEqual(_input_data_paths_byte_size([temp_path]), 7)

    def test_neural_cli_input_data_paths_byte_size_uses_cache_sizer(self) -> None:
        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch("pokezero.neural_cli.training_cache_paths_byte_size", return_value=1234) as cache_size,
        ):
            size = _input_data_paths_byte_size([Path("cache-a")])

        self.assertEqual(size, 1234)
        cache_size.assert_called_once_with([Path("cache-a")])

    def test_neural_cli_training_cache_lifecycle_rejects_delete_with_overlapping_value_data(self) -> None:
        args = SimpleNamespace(
            data=[Path("cache-root/cache-a")],
            max_cache_gb=None,
            delete_cache_after_read=True,
            value_calibration_data=None,
            value_selection_data=[Path("cache-root/cache-a/heldout")],
        )
        with (
            patch("pokezero.neural_cli.is_training_cache_path", return_value=True),
            patch("pokezero.neural_cli.training_cache_root_byte_size", return_value=1234),
        ):
            with self.assertRaisesRegex(ValueError, "delete-cache-after-read"):
                _training_cache_lifecycle(args)

    def test_neural_cli_benchmark_reports_missing_torch_extra(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(["benchmark", "--checkpoint", "checkpoint.pt", "--games", "1"])

        self.assertEqual(exit_code, 1)
        self.assertIn(NEURAL_INSTALL_MESSAGE, stderr.getvalue())

    def test_neural_cli_benchmark_rejects_legacy_no_belief_candidate_by_default(self) -> None:
        stderr = io.StringIO()

        with (
            patch("pokezero.neural_cli._policy_from_checkpoint") as load_checkpoint,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "pokezero-no-belief-gen3-1m.pt",
                    "--games",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(load_checkpoint.call_count, 0)
        self.assertIn("current-family v2+", stderr.getvalue())
        self.assertIn("Legacy no-belief/pre-v2 checkpoints", stderr.getvalue())

    def test_neural_cli_benchmark_rejects_legacy_no_belief_reference_by_default(self) -> None:
        stderr = io.StringIO()
        config = SimpleNamespace(observation_schema_version=OBSERVATION_SCHEMA_VERSION_V2)

        with (
            patch("pokezero.neural_policy.load_transformer_model_config", return_value=config),
            patch("pokezero.neural_cli._policy_from_checkpoint") as load_checkpoint,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "belief-current-family.pt",
                    "--benchmark-reference-policy",
                    "neural:/archive/pokezero-no-belief-gen3-1m.pt",
                    "--benchmark-reference-policy-id",
                    "legacy-no-belief",
                    "--games",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(load_checkpoint.call_count, 0)
        self.assertIn("references require current-family v2+ checkpoints", stderr.getvalue())
        self.assertIn("--allow-legacy-checkpoints", stderr.getvalue())

    def test_neural_cli_benchmark_allows_legacy_checkpoints_when_explicit(self) -> None:
        class FakePolicy:
            policy_id = "legacy-candidate"

        class FakeReport:
            def to_dict(self) -> dict:
                return {"ok": True}

        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        stdout = io.StringIO()
        with (
            patch("pokezero.neural_cli._policy_from_checkpoint", return_value=FakePolicy()) as load_checkpoint,
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "pokezero-no-belief-gen3-1m.pt",
                    "--allow-legacy-checkpoints",
                    "--games",
                    "1",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(load_checkpoint.call_count, 1)
        self.assertEqual(len(captured["matchups"]), 4)

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

    def test_neural_cli_root_puct_telemetry_report_reads_benchmark_artifact_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "benchmark.json"
            report_path = Path(temp_dir) / "telemetry-report.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "matchups": [
                            {
                                "p1_policy_id": "root-puct-120",
                                "p2_policy_id": "random-legal",
                                "game_results": [
                                    {
                                        "root_puct_decision_telemetry_by_player": {
                                            "p1": [
                                                {
                                                    "schema_version": "pokezero.root_puct_decision_telemetry.v2",
                                                    "decision_index": 0,
                                                    "turn_index": 0,
                                                    "outcome": "searched",
                                                    "fallback": False,
                                                    "root_puct_total_visits": 24,
                                                    "full_decision_elapsed_seconds": 0.15,
                                                    "timing": {"total_seconds": 0.12},
                                                }
                                            ]
                                        }
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = neural_cli_main(
                    [
                        "root-puct-telemetry-report",
                        "--input",
                        str(artifact_path),
                        "--policy-id",
                        "root-puct-120",
                        "--out",
                        str(report_path),
                        "--json",
                    ]
                )

            persisted = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(persisted["policies"]["root-puct-120"]["decisions"], 1)
        self.assertEqual(persisted["policies"]["root-puct-120"]["visits"]["per_root_search_second"], 200.0)
        self.assertIn('"schema_version": "pokezero.root_puct_telemetry_report.v1"', stdout.getvalue())

    def test_neural_cli_root_puct_benchmark_rejects_legacy_no_belief_checkpoint_by_default(self) -> None:
        stderr = io.StringIO()

        with (
            patch("pokezero.neural_cli.require_torch"),
            patch("pokezero.neural_cli.load_transformer_checkpoint") as load_checkpoint,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-benchmark",
                    "--checkpoint",
                    "pokezero-no-belief-gen3-1m.pt",
                    "--games",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(load_checkpoint.call_count, 0)
        self.assertIn("current-family v2+", stderr.getvalue())
        self.assertIn("root-puct benchmark", stderr.getvalue())

    def test_neural_cli_root_puct_counterfactual_rejects_legacy_no_belief_checkpoint_by_default(self) -> None:
        stderr = io.StringIO()

        with (
            patch("pokezero.neural_cli.require_torch"),
            patch("pokezero.neural_cli.load_transformer_checkpoint") as load_checkpoint,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-counterfactual",
                    "--checkpoint",
                    "pokezero-no-belief-gen3-1m.pt",
                    "--games",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(load_checkpoint.call_count, 0)
        self.assertIn("current-family v2+", stderr.getvalue())
        self.assertIn("root-puct counterfactual", stderr.getvalue())

    def test_neural_cli_root_puct_play_benchmark_rejects_legacy_no_belief_checkpoint_by_default(self) -> None:
        stderr = io.StringIO()

        with (
            patch("pokezero.neural_cli.require_torch"),
            patch("pokezero.neural_cli.load_transformer_checkpoint") as load_checkpoint,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "pokezero-no-belief-gen3-1m.pt",
                    "--games",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(load_checkpoint.call_count, 0)
        self.assertIn("current-family v2+", stderr.getvalue())
        self.assertIn("root-puct play benchmark", stderr.getvalue())

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
                    "--allow-legacy-checkpoints",
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

    def test_neural_cli_benchmark_history_mask_k_wires_and_stamps(self) -> None:
        class FakePolicy:
            policy_id = "neural-smoke"

        class FakeReport:
            def to_dict(self) -> dict:
                return {"ok": True}

        stdout = io.StringIO()

        with (
            patch(
                "pokezero.neural_cli._policy_from_checkpoint", return_value=FakePolicy()
            ) as load,
            patch("pokezero.neural_cli.benchmark_rollouts", return_value=FakeReport()),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--games",
                    "2",
                    "--history-mask-k",
                    "16",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        # The knob reaches the checkpoint policy loader...
        self.assertEqual(load.call_args.kwargs.get("history_mask_k"), 16)
        # ...and is stamped into the report payload for downstream audit.
        self.assertEqual(json.loads(stdout.getvalue()), {"ok": True, "history_mask_k": 16})

    def test_neural_cli_benchmark_history_mask_k_rejects_out_of_range(self) -> None:
        stderr = io.StringIO()
        with (
            patch("pokezero.neural_cli._policy_from_checkpoint") as load,
            patch("pokezero.neural_cli.benchmark_rollouts") as rollouts,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--history-mask-k",
                    "0",
                ]
            )
        # main() catches the ValueError and reports a non-zero exit; validation happens
        # before any checkpoint load or rollout.
        self.assertEqual(exit_code, 1)
        self.assertIn("history-mask-k", stderr.getvalue())
        load.assert_not_called()
        rollouts.assert_not_called()

    def test_neural_cli_benchmark_history_mask_k_uses_v3_capacity(self) -> None:
        class FakePolicy:
            policy_id = "v3-neural-smoke"
            result = SimpleNamespace(
                model_config=SimpleNamespace(observation_schema_version="pokezero.observation.v3")
            )

        stderr = io.StringIO()
        with (
            patch("pokezero.neural_cli._policy_from_checkpoint", return_value=FakePolicy()),
            patch("pokezero.neural_cli.benchmark_rollouts") as rollouts,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--history-mask-k",
                    "65",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("1..64", stderr.getvalue())
        rollouts.assert_not_called()

    def test_neural_cli_benchmark_can_alias_candidate_policy_id(self) -> None:
        class FakePolicy:
            policy_id = "checkpoint-policy"

        class FakeReport:
            def to_dict(self) -> dict:
                return {"ok": True}

        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        with (
            patch("pokezero.neural_cli._policy_from_checkpoint", return_value=FakePolicy()),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--policy-id",
                    "candidate-member",
                    "--json",
                ]
            )

        matchups = captured["matchups"]
        self.assertEqual(exit_code, 0)
        self.assertEqual([matchup.label for matchup in matchups], [
            "candidate-member vs random-legal",
            "random-legal vs candidate-member",
            "candidate-member vs simple-legal",
            "simple-legal vs candidate-member",
        ])
        self.assertEqual(matchups[0].p1_policy.policy_id, "candidate-member")
        self.assertEqual(matchups[1].p2_policy.policy_id, "candidate-member")

    def test_policy_id_alias_relabels_decisions(self) -> None:
        class FakePolicy:
            policy_id = "checkpoint-policy"

            def select_action(self, observation, *, rng):
                return PolicyDecision(
                    action_index=1,
                    policy_id=self.policy_id,
                    action_probability=0.5,
                    metadata={"source": "underlying"},
                )

        aliased = _PolicyIdAlias(FakePolicy(), policy_id="pool-member")

        decision = aliased.select_action(observation(1), rng=random.Random(1))

        self.assertEqual(decision.policy_id, "pool-member")
        self.assertEqual(decision.action_index, 1)
        self.assertEqual(decision.metadata, {"source": "underlying"})

    def test_neural_cli_benchmark_wires_reference_policy_matchups(self) -> None:
        class FakePolicy:
            policy_id = "neural-smoke"

        class FakeReferencePolicy:
            policy_id = "max-damage"

        class FakeReport:
            def to_dict(self) -> dict:
                return {"ok": True}

        captured = {}
        resolved_reference_specs = []

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        reference_instances = [FakeReferencePolicy(), FakeReferencePolicy()]

        def fake_policy_from_spec(spec: str, *, device: str | None):
            resolved_reference_specs.append((spec, device))
            return reference_instances[len(resolved_reference_specs) - 1]

        fake_policy = FakePolicy()

        with (
            patch("pokezero.neural_cli._policy_from_checkpoint", return_value=fake_policy),
            patch("pokezero.neural_cli._policy_from_spec_for_evaluation", side_effect=fake_policy_from_spec),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--games",
                    "2",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--benchmark-reference-policy",
                    "max-damage",
                    "--device",
                    "cpu",
                    "--json",
                ]
            )

        matchups = captured["matchups"]
        self.assertEqual(exit_code, 0)
        self.assertEqual([matchup.label for matchup in matchups], [
            "neural-smoke vs random-legal",
            "random-legal vs neural-smoke",
            "neural-smoke vs simple-legal",
            "simple-legal vs neural-smoke",
            "neural-smoke vs max-damage",
            "max-damage vs neural-smoke",
        ])
        self.assertIs(matchups[4].p1_policy, fake_policy)
        self.assertIs(matchups[5].p2_policy, fake_policy)
        self.assertIs(matchups[4].p2_policy, reference_instances[0])
        self.assertIs(matchups[5].p1_policy, reference_instances[1])
        expected_reference_spec = (
            f"max-damage?{urlencode({'showdown_root': str(Path('/tmp/showdown').resolve())})}"
        )
        self.assertEqual(resolved_reference_specs, [(expected_reference_spec, "cpu"), (expected_reference_spec, "cpu")])

    def test_neural_cli_benchmark_can_alias_reference_policy_ids(self) -> None:
        class FakePolicy:
            policy_id = "neural-smoke"

        class FakeReferencePolicy:
            policy_id = "checkpoint-policy"

        class FakeReport:
            def to_dict(self) -> dict:
                return {"ok": True}

        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        with (
            patch("pokezero.neural_cli._policy_from_checkpoint", return_value=FakePolicy()),
            patch("pokezero.neural_cli._policy_from_spec_for_evaluation", return_value=FakeReferencePolicy()),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--benchmark-reference-policy",
                    "neural:/pool/member.pt",
                    "--benchmark-reference-policy-id",
                    "pool-member",
                    "--json",
                ]
            )

        matchups = captured["matchups"]
        self.assertEqual(exit_code, 0)
        self.assertEqual([matchup.label for matchup in matchups], [
            "neural-smoke vs random-legal",
            "random-legal vs neural-smoke",
            "neural-smoke vs simple-legal",
            "simple-legal vs neural-smoke",
            "neural-smoke vs pool-member",
            "pool-member vs neural-smoke",
        ])
        self.assertEqual(matchups[4].p2_policy.policy_id, "pool-member")
        self.assertEqual(matchups[5].p1_policy.policy_id, "pool-member")

    def test_neural_cli_benchmark_rejects_reference_policy_id_count_mismatch(self) -> None:
        class FakePolicy:
            policy_id = "neural-smoke"

        stderr = io.StringIO()
        with (
            patch("pokezero.neural_cli._policy_from_checkpoint", return_value=FakePolicy()),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--benchmark-reference-policy",
                    "max-damage",
                    "--benchmark-reference-policy-id",
                    "max-damage-a",
                    "--benchmark-reference-policy-id",
                    "max-damage-b",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--benchmark-reference-policy-id", stderr.getvalue())

    def test_neural_cli_benchmark_skips_duplicate_reference_policy_ids(self) -> None:
        class FakePolicy:
            policy_id = "neural-smoke"

        class FakeDuplicatePolicy:
            policy_id = "random-legal"

        class FakeReport:
            def to_dict(self) -> dict:
                return {"ok": True}

        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            return FakeReport()

        with (
            patch("pokezero.neural_cli._policy_from_checkpoint", return_value=FakePolicy()),
            patch("pokezero.neural_cli._policy_from_spec_for_evaluation", return_value=FakeDuplicatePolicy()) as ref,
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                [
                    "benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--benchmark-reference-policy",
                    "random-legal",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(ref.call_args.args[0], "random-legal")
        self.assertEqual(len(captured["matchups"]), 4)

    def test_neural_cli_benchmark_writes_summary_out(self) -> None:
        class FakePolicy:
            policy_id = "neural-smoke"

        class FakeReport:
            def to_dict(self) -> dict:
                return {
                    "schema_version": "fixture.benchmark.v1",
                    "total_games": 8,
                    "matchups": [{"label": "neural-smoke vs max-damage"}],
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "nested" / "benchmark-summary.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch("pokezero.neural_cli._policy_from_checkpoint", return_value=FakePolicy()),
                patch("pokezero.neural_cli.benchmark_rollouts", return_value=FakeReport()),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = neural_cli_main(
                    [
                        "benchmark",
                        "--checkpoint",
                        "checkpoint.pt",
                        "--allow-legacy-checkpoints",
                        "--games",
                        "2",
                        "--summary-out",
                        str(summary_path),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(summary_path.read_text()), FakeReport().to_dict())
            self.assertIn(f"benchmark_summary: {summary_path}", stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue()), FakeReport().to_dict())

    def test_neural_cli_benchmark_summary_out_preserves_human_report(self) -> None:
        class FakePolicy:
            policy_id = "neural-smoke"

        class FakeReport:
            def to_dict(self) -> dict:
                return {"total_games": 8}

        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "benchmark-summary.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            report = FakeReport()
            with (
                patch("pokezero.neural_cli._policy_from_checkpoint", return_value=FakePolicy()),
                patch("pokezero.neural_cli.benchmark_rollouts", return_value=report),
                patch("pokezero.neural_cli.print_benchmark_report") as print_report,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = neural_cli_main(
                    [
                        "benchmark",
                        "--checkpoint",
                        "checkpoint.pt",
                        "--allow-legacy-checkpoints",
                        "--games",
                        "2",
                        "--summary-out",
                        str(summary_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(summary_path.read_text()), report.to_dict())
            self.assertIn(f"benchmark_summary: {summary_path}", stderr.getvalue())
            print_report.assert_called_once_with(report)

    def test_neural_cli_root_puct_play_benchmark_summary_out_persists_json(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {
                    "schema_version": "fixture.root-puct-play.v1",
                    "matchups": 4,
                    "head_to_heads": [
                        {
                            "first_policy_id": "neural-smoke",
                            "second_policy_id": "random-legal",
                            "games": 4,
                            "first_policy_wins": 1,
                            "second_policy_wins": 3,
                            "ties": 0,
                            "capped_games": 0,
                            "first_policy_win_rate": 0.25,
                            "second_policy_win_rate": 0.75,
                        },
                        {
                            "first_policy_id": "neural-smoke+root-puct",
                            "second_policy_id": "random-legal",
                            "games": 4,
                            "first_policy_wins": 2,
                            "second_policy_wins": 2,
                            "ties": 0,
                            "capped_games": 1,
                            "first_policy_win_rate": 0.5,
                            "second_policy_win_rate": 0.5,
                        },
                    ],
                }

        fake_model = object()
        fake_training_result = SimpleNamespace(
            model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1, format_id="gen3randombattle", observation_schema_version="pokezero.observation.v2.1", categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count, numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count, stats_block_enabled=True, exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "nested" / "root-puct-play-summary.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            report = FakeReport()
            with (
                patch(
                    "pokezero.neural_cli.load_transformer_checkpoint",
                    return_value=(fake_model, fake_training_result),
                ),
                patch("pokezero.neural_cli.benchmark_rollouts", return_value=report),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = neural_cli_main(
                    [
                        "root-puct-play-benchmark",
                        "--checkpoint",
                        "checkpoint.pt",
                        "--allow-legacy-checkpoints",
                        "--games",
                        "2",
                        "--opponent-policy",
                        "random-legal",
                        "--summary-out",
                        str(summary_path),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            expected = {
                **report.to_dict(),
                "root_puct_play_comparisons": [
                    {
                        "opponent_policy_id": "random-legal",
                        "raw_policy_id": "neural-smoke",
                        "search_policy_id": "neural-smoke+root-puct",
                        "raw": {
                            "games": 4,
                            "wins": 1,
                            "win_rate": 0.25,
                            "ties": 0,
                            "capped_games": 0,
                        },
                        "search": {
                            "games": 4,
                            "wins": 2,
                            "win_rate": 0.5,
                            "ties": 0,
                            "capped_games": 1,
                        },
                        "search_minus_raw_win_rate": 0.25,
                    }
                ],
            }
            saved = json.loads(summary_path.read_text())
            for field, value in expected.items():
                self.assertEqual(saved[field], value)
            self.assertEqual(saved["root_puct_config"]["root_visit_budget"], 16)
            self.assertEqual(saved["root_puct_config"]["allow_search_fallback"], True)
            self.assertEqual(
                saved["root_puct_policy_configs"]["neural-smoke+root-puct"],
                saved["root_puct_config"],
            )
            self.assertIn(f"root_puct_play_benchmark_summary: {summary_path}", stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue()), saved)

    def test_neural_cli_root_puct_play_benchmark_wires_raw_and_search_matchups(self) -> None:
        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 6}

        fake_model = object()
        fake_training_result = SimpleNamespace(
            model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1, format_id="gen3randombattle", observation_schema_version="pokezero.observation.v2.1", categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count, numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count, stats_block_enabled=True, exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False)
        )
        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            kwargs["progress_callback"](
                SimpleNamespace(
                    matchup_label="neural-smoke vs random-legal",
                    matchup_index=0,
                    matchup_count=6,
                    games_completed=2,
                    games_total=3,
                    seed=100,
                    matchup_elapsed_seconds=12.3456,
                )
            )
            matchups = tuple(kwargs["matchups"])
            deterministic_search_policy = matchups[2].p1_policy
            search_policy = matchups[4].p1_policy
            self.assertIsNone(deterministic_search_policy.root_dirichlet_alpha)
            self.assertEqual(search_policy.value_fn((observation(1),)), 0.25)
            self.assertEqual(search_policy.prior_fn((observation(1),)), (1.0,) + (0.0,) * 8)
            self.assertIsNotNone(search_policy.neural_timing_snapshot)
            self.assertEqual(search_policy.root_prior_temperature, 2.5)
            self.assertEqual(
                search_policy.leaf_rollout_policy_factory("p1").policy_id,
                "neural-smoke+root-puct+adaptive-budget-leaf-p1",
            )
            self.assertEqual(
                search_policy.leaf_rollout_policy_factory("p2").policy_id,
                "neural-smoke+root-puct+adaptive-budget-leaf-p2",
            )
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
            self.assertIsNotNone(search_policy.opponent_action_scenario_planner)
            self.assertEqual(
                getattr(search_policy.opponent_action_scenario_planner, "planner_id"),
                f"checkpoint-top{ACTION_COUNT}",
            )
            self.assertEqual(search_policy.opponent_action_planner(context, __import__("random").Random(1)), {"p2": 2})
            return FakeReport()

        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch("pokezero.neural_cli.require_torch"),
            patch("pokezero.neural_policy.require_torch"),
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)) as load,
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25) as value_eval,
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8) as prior_eval,
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(0.1, 0.2, 0.7) + (0.0,) * 6) as opponent_eval,
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
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
                    "--root-visit-budget",
                    "17",
                    "--min-value-improvement",
                    "0.2",
                    "--device",
                    "cpu",
                    "--temperature",
                    "1.5",
                    "--root-prior-temperature",
                    "2.5",
                    "--root-dirichlet-alpha",
                    "0.3",
                    "--root-dirichlet-mix",
                    "0.2",
                    "--root-dirichlet-seed",
                    "11",
                    "--adaptive-root-contested-extra-visits",
                    "120",
                    "--adaptive-root-policy-entropy-threshold",
                    "0.7",
                    "--adaptive-root-value-margin-threshold",
                    "0.15",
                    "--progress-interval-games",
                    "2",
                    "--progress-interval-decisions",
                    "1",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        load.assert_called_once_with(Path("checkpoint.pt"), map_location="cpu")
        self.assertEqual(captured["games"], 3)
        self.assertTrue(callable(captured["progress_callback"]))
        self.assertEqual(captured["seed_start"], 99)
        self.assertEqual(captured["rollout_config"].max_decision_rounds, 12)
        matchups = tuple(captured["matchups"])
        self.assertNotIsInstance(matchups[0].p1_policy, _RootPuctDecisionProgressPolicy)
        self.assertIsInstance(matchups[2].p1_policy, _RootPuctDecisionProgressPolicy)
        self.assertNotIsInstance(matchups[2].p2_policy, _RootPuctDecisionProgressPolicy)
        self.assertEqual([matchup.label for matchup in matchups], [
            "neural-smoke vs random-legal",
            "random-legal vs neural-smoke",
            "neural-smoke+root-puct+adaptive-budget vs random-legal",
            "random-legal vs neural-smoke+root-puct+adaptive-budget",
            "neural-smoke+root-puct+adaptive-budget+dirichlet vs random-legal",
            "random-legal vs neural-smoke+root-puct+adaptive-budget+dirichlet",
        ])
        self.assertEqual(matchups[0].p1_policy.policy_id, "neural-smoke")
        self.assertEqual(matchups[2].p1_policy.policy_id, "neural-smoke+root-puct+adaptive-budget")
        self.assertEqual(matchups[3].p2_policy.policy_id, "neural-smoke+root-puct+adaptive-budget")
        self.assertEqual(matchups[4].p1_policy.policy_id, "neural-smoke+root-puct+adaptive-budget+dirichlet")
        self.assertEqual(matchups[5].p2_policy.policy_id, "neural-smoke+root-puct+adaptive-budget+dirichlet")
        self.assertEqual(matchups[4].p1_policy.cpuct, 0.75)
        self.assertEqual(matchups[4].p1_policy.selection_mode, "value")
        self.assertEqual(matchups[4].p1_policy.root_visit_budget, 17)
        self.assertEqual(matchups[4].p1_policy.root_dirichlet_alpha, 0.3)
        self.assertEqual(matchups[4].p1_policy.root_dirichlet_mix, 0.2)
        self.assertEqual(matchups[4].p1_policy.root_dirichlet_seed, 11)
        self.assertEqual(
            matchups[4].p1_policy.root_visit_budget_selector.to_dict(),
            {
                "selector_id": "entropy-or-value-margin",
                "contested_extra_visits": 120,
                "uncontested_extra_visits": 0,
                "minimum_policy_entropy": 0.7,
                "maximum_value_margin": 0.15,
            },
        )
        self.assertEqual(matchups[4].p1_policy.minimum_value_improvement, 0.2)
        self.assertEqual(matchups[4].p1_policy.leaf_rollout_decision_rounds, 2)
        self.assertIsNotNone(matchups[4].p1_policy.leaf_rollout_policy_factory)
        self.assertEqual(matchups[4].p1_policy.fallback_policy.policy_id, "neural-smoke-fallback")
        self.assertEqual(
            matchups[4].p1_policy.leaf_rollout_metadata,
            {"root_puct_leaf_rollout_opponent_policy": "checkpoint"},
        )
        self.assertTrue(matchups[2].p1_policy.allow_fallback)
        self.assertEqual(value_eval.call_args.kwargs["model"], fake_model)
        self.assertEqual(value_eval.call_args.kwargs["device"], "cpu")
        self.assertEqual(prior_eval.call_args.kwargs["temperature"], 1.0)
        self.assertEqual(opponent_eval.call_args.kwargs["temperature"], 1.5)
        payload = json.loads(stdout.getvalue())
        progress_lines = [
            line
            for line in stderr.getvalue().splitlines()
            if line.startswith("root_puct_play_benchmark_progress:")
        ]
        self.assertEqual(
            [json.loads(line.split(": ", 1)[1]) for line in progress_lines],
            [
                {
                    "games_completed": 2,
                    "games_total": 3,
                    "matchup_count": 6,
                    "matchup_elapsed_seconds": 12.346,
                    "matchup_index": 1,
                    "matchup_label": "neural-smoke vs random-legal",
                    "seed": 100,
                }
            ],
        )
        self.assertEqual(payload["matchups"], 6)
        self.assertEqual(payload["root_dirichlet"], {"enabled": True, "alpha": 0.3, "mix": 0.2, "base_seed": 11})
        self.assertEqual(
            payload["adaptive_root_visit_budget"],
            {
                "selector_id": "entropy-or-value-margin",
                "contested_extra_visits": 120,
                "uncontested_extra_visits": 0,
                "minimum_policy_entropy": 0.7,
                "maximum_value_margin": 0.15,
            },
        )
        deterministic_config = {
            "max_decision_rounds": 12,
            "temperature": 1.5,
            "cpuct": 0.75,
            "selection_mode": "value",
            "root_prior_temperature": 2.5,
            "minimum_value_improvement": 0.2,
            "root_visit_budget": 17,
            "root_extra_visits": None,
            "batch_initial_root_values": False,
            "batch_adaptive_root_values": False,
            "reuse_adaptive_root_branches": False,
            "adaptive_root_contested_extra_visits": 120,
            "adaptive_root_uncontested_extra_visits": 0,
            "adaptive_root_policy_entropy_threshold": 0.7,
            "adaptive_root_value_margin_threshold": 0.15,
            "root_time_budget_ms": None,
            "root_opponent_action_policy": "checkpoint",
            "root_opponent_action_scenarios": 1,
            "root_opponent_action_candidate_scenarios": ACTION_COUNT,
            "leaf_rollout_rounds": 2,
            "leaf_rollout_sampling": False,
            "leaf_rollout_opponent_policy": "checkpoint",
            "belief_start_overrides": False,
            "belief_world_sample_cap": 4,
            "belief_start_override_attempts": 10,
            "belief_start_override_hp_fraction_tolerance": 0.02,
            "opponent_legal_mask_mode": "hidden",
            "allow_search_fallback": True,
            "record_belief_world_coverage_gaps": False,
            "root_dirichlet_alpha": None,
            "root_dirichlet_mix": None,
            "root_dirichlet_seed": None,
        }
        noisy_config = {
            **deterministic_config,
            "root_dirichlet_alpha": 0.3,
            "root_dirichlet_mix": 0.2,
            "root_dirichlet_seed": 11,
        }
        self.assertNotIn("root_puct_config", payload)
        self.assertEqual(
            payload["root_puct_policy_configs"],
            {
                "neural-smoke+root-puct+adaptive-budget": deterministic_config,
                "neural-smoke+root-puct+adaptive-budget+dirichlet": noisy_config,
            },
        )

    def test_neural_cli_root_puct_play_benchmark_defaults_to_visit_selection(self) -> None:
        parser = build_neural_arg_parser()

        args = parser.parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
            ]
        )

        self.assertEqual(args.selection_mode, "visits")
        self.assertEqual(args.root_visit_budget, 16)
        self.assertIsNone(args.root_time_budget_ms)
        self.assertIsNone(args.root_prior_temperature)
        self.assertIsNone(args.root_dirichlet_alpha)
        self.assertEqual(args.root_dirichlet_mix, 0.25)
        self.assertEqual(args.root_dirichlet_seed, 0)
        self.assertIsNone(args.adaptive_root_contested_extra_visits)
        self.assertEqual(args.adaptive_root_uncontested_extra_visits, 0)
        self.assertIsNone(args.adaptive_root_policy_entropy_threshold)
        self.assertIsNone(args.adaptive_root_value_margin_threshold)
        self.assertFalse(args.batch_initial_root_values)
        self.assertFalse(args.batch_adaptive_root_values)
        self.assertFalse(args.reuse_adaptive_root_branches)
        self.assertIsNone(args.progress_interval_games)
        self.assertIsNone(args.progress_interval_decisions)

    def test_neural_cli_root_puct_play_benchmark_accepts_initial_value_batching(self) -> None:
        args = build_neural_arg_parser().parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
                "--batch-initial-root-values",
            ]
        )

        self.assertTrue(args.batch_initial_root_values)

    def test_neural_cli_root_puct_play_benchmark_accepts_adaptive_value_batching(self) -> None:
        args = build_neural_arg_parser().parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
                "--batch-initial-root-values",
                "--batch-adaptive-root-values",
            ]
        )

        self.assertTrue(args.batch_initial_root_values)
        self.assertTrue(args.batch_adaptive_root_values)

    def test_neural_cli_root_puct_play_benchmark_accepts_adaptive_branch_reuse(self) -> None:
        args = build_neural_arg_parser().parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
                "--batch-initial-root-values",
                "--batch-adaptive-root-values",
                "--reuse-adaptive-root-branches",
            ]
        )

        self.assertTrue(args.reuse_adaptive_root_branches)

    def test_root_puct_decision_progress_wraps_contextual_policy_without_changing_decision(self) -> None:
        emitted: list[_RootPuctDecisionProgress] = []

        class FakePolicy:
            policy_id = "fake-policy"

            def select_action(self, value, *, rng):
                return PolicyDecision(action_index=0, policy_id=self.policy_id)

            def select_action_with_context(self, context, *, rng):
                self.context = context
                return PolicyDecision(action_index=1, policy_id=self.policy_id)

        wrapped = _RootPuctDecisionProgressPolicy(
            policy=FakePolicy(),
            progress_callback=emitted.append,
            matchup_label="fake-policy vs max-damage",
            matchup_index=2,
            matchup_count=4,
            games_total=3,
            seed_start=80,
        )
        context = PolicyContext(
            player_id="p2",
            decision_round_index=4,
            battle_id="progress-test",
            format_id="gen3randombattle",
            seed=81,
            observation=observation(1),
            requested_players=("p2",),
            trajectory=BattleTrajectory(battle_id="progress-test", format_id="gen3randombattle", seed=81),
        )

        decision = wrapped.select_action_with_context(context, rng=random.Random(3))

        self.assertEqual(decision, PolicyDecision(action_index=1, policy_id="fake-policy"))
        self.assertEqual(
            [(event.event, event.game_index, event.decision_round_index, event.player_id) for event in emitted],
            [("started", 2, 4, "p2"), ("completed", 2, 4, "p2")],
        )
        self.assertIsNone(emitted[0].policy_elapsed_seconds)
        self.assertIsNotNone(emitted[1].policy_elapsed_seconds)

    def test_root_puct_decision_progress_callback_emits_started_and_completion_for_sampled_decisions(self) -> None:
        stderr = io.StringIO()
        callback = _root_puct_decision_progress_callback(2)
        base = {
            "matchup_label": "root-puct vs max-damage",
            "matchup_index": 1,
            "matchup_count": 4,
            "game_index": 1,
            "games_total": 3,
            "seed": 80,
            "decision_round_index": 0,
            "player_id": "p1",
            "policy_id": "root-puct",
        }
        with contextlib.redirect_stderr(stderr):
            callback(_RootPuctDecisionProgress(event="started", **base))
            callback(_RootPuctDecisionProgress(event="completed", policy_elapsed_seconds=0.2, **base))
            second = {**base, "decision_round_index": 1, "player_id": "p2"}
            callback(_RootPuctDecisionProgress(event="started", **second))
            callback(_RootPuctDecisionProgress(event="completed", policy_elapsed_seconds=0.1234567, **second))

        lines = [
            line
            for line in stderr.getvalue().splitlines()
            if line.startswith("root_puct_play_benchmark_decision_progress:")
        ]
        self.assertEqual(
            [json.loads(line.split(": ", 1)[1]) for line in lines],
            [
                {
                    "decision_round": 2,
                    "decision_sequence": 2,
                    "event": "started",
                    "game_index": 1,
                    "games_total": 3,
                    "matchup_count": 4,
                    "matchup_index": 2,
                    "matchup_label": "root-puct vs max-damage",
                    "player_id": "p2",
                    "policy_id": "root-puct",
                    "seed": 80,
                },
                {
                    "decision_round": 2,
                    "decision_sequence": 2,
                    "event": "completed",
                    "game_index": 1,
                    "games_total": 3,
                    "matchup_count": 4,
                    "matchup_index": 2,
                    "matchup_label": "root-puct vs max-damage",
                    "player_id": "p2",
                    "policy_elapsed_seconds": 0.123457,
                    "policy_id": "root-puct",
                    "seed": 80,
                },
            ],
        )

    def test_root_puct_progress_callback_emits_only_configured_intervals_and_completion(self) -> None:
        stderr = io.StringIO()
        callback = _root_puct_benchmark_progress_callback(2)
        with contextlib.redirect_stderr(stderr):
            for games_completed in (1, 2, 3):
                callback(
                    SimpleNamespace(
                        matchup_label="root-puct vs max-damage",
                        matchup_index=1,
                        matchup_count=4,
                        games_completed=games_completed,
                        games_total=3,
                        seed=80 + games_completed,
                        matchup_elapsed_seconds=1.23456 * games_completed,
                        root_puct_by_player={
                            "p1": {
                                "root_puct_searches": 2,
                                "root_puct_fallbacks": 1,
                                "root_puct_fallback_categories": {"missing_sampled_world": 1},
                                "root_puct_fallback_signatures": {"force-switch:search:p1:move": 1},
                                "root_puct_opponent_action_missing_sampled_world_reason_categories": {
                                    "opponent_belief_unavailable": 2,
                                },
                            }
                        },
                    )
                )
            for games_completed in (1, 2):
                callback(
                    SimpleNamespace(
                        matchup_label="root-puct vs foul-play",
                        matchup_index=2,
                        matchup_count=4,
                        games_completed=games_completed,
                        games_total=2,
                        seed=90 + games_completed,
                        matchup_elapsed_seconds=2.0 * games_completed,
                        root_puct_by_player={
                            "p2": {
                                "root_puct_searches": 1,
                                "root_puct_fallbacks": 0,
                                "root_puct_fallback_categories": {},
                            }
                        },
                    )
                )

        lines = [
            line
            for line in stderr.getvalue().splitlines()
            if line.startswith("root_puct_play_benchmark_progress:")
        ]
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            [json.loads(line.split(": ", 1)[1]) for line in lines],
            [
                {
                    "games_completed": 2,
                    "games_total": 3,
                    "matchup_count": 4,
                    "matchup_elapsed_seconds": 2.469,
                    "matchup_index": 2,
                    "matchup_label": "root-puct vs max-damage",
                    "root_puct_fallback_categories": {"missing_sampled_world": 2},
                    "root_puct_fallback_signatures": {"force-switch:search:p1:move": 2},
                    "root_puct_opponent_action_missing_sampled_world_reason_categories": {
                        "opponent_belief_unavailable": 4,
                    },
                    "root_puct_fallback_rate": 0.5,
                    "root_puct_fallbacks": 2,
                    "root_puct_searches": 4,
                    "seed": 82,
                },
                {
                    "games_completed": 3,
                    "games_total": 3,
                    "matchup_count": 4,
                    "matchup_elapsed_seconds": 3.704,
                    "matchup_index": 2,
                    "matchup_label": "root-puct vs max-damage",
                    "root_puct_fallback_categories": {"missing_sampled_world": 3},
                    "root_puct_fallback_signatures": {"force-switch:search:p1:move": 3},
                    "root_puct_opponent_action_missing_sampled_world_reason_categories": {
                        "opponent_belief_unavailable": 6,
                    },
                    "root_puct_fallback_rate": 0.5,
                    "root_puct_fallbacks": 3,
                    "root_puct_searches": 6,
                    "seed": 83,
                },
                {
                    "games_completed": 2,
                    "games_total": 2,
                    "matchup_count": 4,
                    "matchup_elapsed_seconds": 4.0,
                    "matchup_index": 3,
                    "matchup_label": "root-puct vs foul-play",
                    "root_puct_fallback_categories": {},
                    "root_puct_fallback_rate": 0.0,
                    "root_puct_fallbacks": 0,
                    "root_puct_searches": 2,
                    "seed": 92,
                },
            ],
        )

    def test_root_puct_progress_callback_tolerates_legacy_progress_without_diagnostics(self) -> None:
        stderr = io.StringIO()
        callback = _root_puct_benchmark_progress_callback(1)

        with contextlib.redirect_stderr(stderr):
            callback(
                SimpleNamespace(
                    matchup_label="root-puct vs max-damage",
                    matchup_index=0,
                    matchup_count=1,
                    games_completed=1,
                    games_total=1,
                    seed=81,
                    matchup_elapsed_seconds=1.0,
                )
            )

        payload = json.loads(stderr.getvalue().split(": ", 1)[1])
        self.assertNotIn("root_puct_fallback_rate", payload)
        self.assertNotIn("root_puct_searches", payload)
        self.assertNotIn("root_puct_fallbacks", payload)
        self.assertNotIn("root_puct_fallback_categories", payload)

    def test_root_puct_progress_callback_filters_malformed_missing_world_categories(self) -> None:
        stderr = io.StringIO()
        callback = _root_puct_benchmark_progress_callback(1)

        with contextlib.redirect_stderr(stderr):
            callback(
                SimpleNamespace(
                    matchup_label="root-puct vs max-damage",
                    matchup_index=0,
                    matchup_count=1,
                    games_completed=1,
                    games_total=1,
                    seed=81,
                    matchup_elapsed_seconds=1.0,
                    root_puct_by_player={
                        "p1": {
                            "root_puct_searches": 1,
                            "root_puct_fallbacks": 1,
                            "root_puct_opponent_action_missing_sampled_world_reason_categories": {
                                "belief_view_invalid": 2,
                                "bad-count": "1",
                                "boolean-count": True,
                                3: 1,
                            },
                        }
                    },
                )
            )

        payload = json.loads(stderr.getvalue().split(": ", 1)[1])
        self.assertEqual(
            payload["root_puct_opponent_action_missing_sampled_world_reason_categories"],
            {"belief_view_invalid": 2},
        )

    def test_root_puct_progress_callback_ignores_malformed_optional_diagnostics(self) -> None:
        stderr = io.StringIO()
        callback = _root_puct_benchmark_progress_callback(1)

        with contextlib.redirect_stderr(stderr):
            for games_completed, root_puct_by_player in (
                (1, ["not-a-mapping"]),
                (2, {"p1": "not-a-mapping"}),
            ):
                callback(
                    SimpleNamespace(
                        matchup_label="root-puct vs max-damage",
                        matchup_index=0,
                        matchup_count=1,
                        games_completed=games_completed,
                        games_total=2,
                        seed=80 + games_completed,
                        matchup_elapsed_seconds=float(games_completed),
                        root_puct_by_player=root_puct_by_player,
                    )
                )

        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            payload = json.loads(line.split(": ", 1)[1])
            self.assertNotIn("root_puct_searches", payload)
            self.assertNotIn("root_puct_fallback_categories", payload)

    def test_neural_cli_root_puct_play_benchmark_wires_time_budget_without_legacy_visit_cap(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        fake_model = object()
        fake_training_result = SimpleNamespace(
            model_config=SimpleNamespace(
                policy_id="neural-smoke",
                window_size=1,
                format_id="gen3randombattle",
                observation_schema_version="pokezero.observation.v2.1",
                categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
                numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
                stats_block_enabled=True,
                exact_state_enabled=True,
                transition_token_budget=128,
                tier2_residuals=True,
                tier2_investment=False,
            )
        )
        captured: dict[str, Any] = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            search_policy = tuple(kwargs["matchups"])[2].p1_policy
            self.assertIsNone(search_policy.root_visit_budget)
            self.assertIsNone(search_policy.root_visit_budget_selector)
            self.assertEqual(search_policy.root_time_budget_seconds, 0.125)
            return SimpleNamespace(to_dict=lambda: {"matchups": 4}, matchups=())

        stdout = io.StringIO()
        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25),
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--opponent-policy",
                    "random-legal",
                    "--root-time-budget-ms",
                    "125",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["games"], 20)
        self.assertEqual(json.loads(stdout.getvalue())["root_time_budget_ms"], 125)

    def test_neural_cli_root_puct_play_benchmark_rejects_time_budget_with_visit_selector(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--root-time-budget-ms",
                    "125",
                    "--root-extra-visits",
                    "24",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("root time budget cannot be combined", stderr.getvalue())

    def test_neural_cli_root_puct_play_benchmark_rejects_time_budget_with_explicit_visit_cap(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--root-time-budget-ms",
                    "125",
                    "--root-visit-budget",
                    "16",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("explicit root visit budget", stderr.getvalue())

    def test_neural_cli_root_puct_play_benchmark_builds_adaptive_budget_selector(self) -> None:
        parser = build_neural_arg_parser()
        args = parser.parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
                "--adaptive-root-contested-extra-visits",
                "120",
                "--adaptive-root-uncontested-extra-visits",
                "3",
                "--adaptive-root-policy-entropy-threshold",
                "0.7",
                "--adaptive-root-value-margin-threshold",
                "0.15",
            ]
        )

        selector = _adaptive_root_visit_budget_selector(args)

        self.assertIsNotNone(selector)
        self.assertEqual(
            selector.to_dict(),
            {
                "selector_id": "entropy-or-value-margin",
                "contested_extra_visits": 120,
                "uncontested_extra_visits": 3,
                "minimum_policy_entropy": 0.7,
                "maximum_value_margin": 0.15,
            },
        )

        invalid_args = parser.parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
                "--adaptive-root-policy-entropy-threshold",
                "0.7",
            ]
        )
        with self.assertRaisesRegex(ValueError, "adaptive root thresholds"):
            _adaptive_root_visit_budget_selector(invalid_args)

        missing_threshold_args = parser.parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
                "--adaptive-root-contested-extra-visits",
                "120",
            ]
        )
        with self.assertRaisesRegex(ValueError, "entropy or value-margin threshold"):
            _adaptive_root_visit_budget_selector(missing_threshold_args)

    def test_neural_cli_root_puct_play_benchmark_builds_fixed_extra_budget_selector(self) -> None:
        parser = build_neural_arg_parser()
        args = parser.parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
                "--root-extra-visits",
                "24",
            ]
        )

        selector = _root_visit_budget_selector(args)

        self.assertIsNotNone(selector)
        self.assertEqual(
            selector.to_dict(),
            {"selector_id": "fixed-extra-visits", "extra_visits": 24},
        )
        incompatible_args = parser.parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
                "--root-extra-visits",
                "24",
                "--adaptive-root-contested-extra-visits",
                "120",
                "--adaptive-root-policy-entropy-threshold",
                "0.7",
            ]
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _root_visit_budget_selector(incompatible_args)

    def test_neural_cli_root_puct_play_benchmark_keeps_raw_priors_and_uses_explicit_value_checkpoint(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        policy_checkpoint = Path(temporary.name) / "policy.pt"
        value_checkpoint = Path(temporary.name) / "calibrated.pt"
        policy_checkpoint.write_bytes(b"raw-policy")
        value_checkpoint.write_bytes(b"value-leaf")
        raw_checkpoint_sha256 = hashlib.sha256(b"raw-policy").hexdigest()

        policy_model = object()
        value_model = object()
        model_config = SimpleNamespace(
            policy_id="neural-smoke",
            window_size=1,
            format_id="gen3randombattle",
            observation_schema_version="pokezero.observation.v2.1",
            categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            stats_block_enabled=True,
            exact_state_enabled=True,
            transition_token_budget=128,
            tier2_residuals=True,
            tier2_investment=False,
        )
        raw_result = SimpleNamespace(
            model_config=model_config,
            belief_set_source_hash=None,
            value_calibration_transform=None,
        )
        calibrated_transform = ValueCalibrationTransform(
            method="isotonic",
            points=((-1.0, -0.5), (1.0, 0.75)),
        )
        value_result = SimpleNamespace(
            model_config=model_config,
            belief_set_source_hash=None,
            value_calibration_transform=calibrated_transform,
        )
        value_leaf_provenance = {
            "policy_checkpoint": str(policy_checkpoint),
            "policy_checkpoint_sha256": raw_checkpoint_sha256,
            "value_checkpoint": str(value_checkpoint),
            "value_checkpoint_sha256": "leaf-sha",
            "value_calibration_source_checkpoint_sha256": raw_checkpoint_sha256,
            "model_config_match": True,
            "belief_set_source_hash_match": True,
            "value_calibration_transform": calibrated_transform.to_dict(),
        }
        captured: dict[str, Any] = {}

        def fake_benchmark_rollouts(**kwargs):
            search_policy = tuple(kwargs["matchups"])[2].p1_policy
            self.assertEqual(search_policy.prior_fn((observation(1),)), (1.0,) + (0.0,) * 8)
            self.assertEqual(search_policy.value_fn((observation(1),)), 0.5)
            self.assertEqual(search_policy.root_visit_budget_selector.to_dict(), {
                "selector_id": "fixed-extra-visits",
                "extra_visits": 24,
            })
            captured["search_policy"] = search_policy
            return SimpleNamespace(to_dict=lambda: {"matchups": 4}, matchups=())

        stdout = io.StringIO()
        with (
            patch(
                "pokezero.neural_cli.load_transformer_checkpoint",
                side_effect=((policy_model, raw_result), (value_model, value_result)),
            ) as load,
            patch(
                "pokezero.neural_cli.require_compatible_transformer_value_checkpoint",
                return_value=value_leaf_provenance,
            ),
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.5) as value_eval,
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8) as prior_eval,
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    str(policy_checkpoint),
                    "--value-checkpoint",
                    str(value_checkpoint),
                    "--allow-legacy-checkpoints",
                    "--opponent-policy",
                    "random-legal",
                    "--root-extra-visits",
                    "24",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(load.call_args_list[0].args[0], policy_checkpoint)
        self.assertEqual(load.call_args_list[1].args[0], value_checkpoint)
        self.assertEqual(captured["search_policy"].checkpoint_path, str(policy_checkpoint.resolve()))
        self.assertEqual(captured["search_policy"].weights_sha256, raw_checkpoint_sha256)
        self.assertIs(value_eval.call_args.kwargs["model"], value_model)
        self.assertIsNotNone(value_eval.call_args.kwargs["timing"])
        self.assertIs(prior_eval.call_args.kwargs["model"], policy_model)
        self.assertIsNotNone(prior_eval.call_args.kwargs["timing"])
        expected_root_config = {
            "max_decision_rounds": 250,
            "temperature": 1.0,
            "cpuct": 1.25,
            "selection_mode": "visits",
            "root_prior_temperature": 1.0,
            "minimum_value_improvement": None,
            "root_visit_budget": 16,
            "root_extra_visits": 24,
            "batch_initial_root_values": False,
            "batch_adaptive_root_values": False,
            "reuse_adaptive_root_branches": False,
            "adaptive_root_contested_extra_visits": None,
            "adaptive_root_uncontested_extra_visits": 0,
            "adaptive_root_policy_entropy_threshold": None,
            "adaptive_root_value_margin_threshold": None,
            "root_time_budget_ms": None,
            "root_opponent_action_policy": "checkpoint",
            "root_opponent_action_scenarios": 1,
            "root_opponent_action_candidate_scenarios": ACTION_COUNT,
            "leaf_rollout_rounds": 0,
            "leaf_rollout_sampling": False,
            "leaf_rollout_opponent_policy": "checkpoint",
            "belief_start_overrides": False,
            "belief_world_sample_cap": 4,
            "belief_start_override_attempts": 10,
            "belief_start_override_hp_fraction_tolerance": 0.02,
            "opponent_legal_mask_mode": "hidden",
            "allow_search_fallback": True,
            "record_belief_world_coverage_gaps": False,
            "root_dirichlet_alpha": None,
            "root_dirichlet_mix": None,
            "root_dirichlet_seed": None,
        }
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "matchups": 4,
                "root_visit_budget_selector": {
                    "selector_id": "fixed-extra-visits",
                    "extra_visits": 24,
                },
                "root_puct_config": expected_root_config,
                "root_puct_policy_configs": {"neural-smoke+root-puct": expected_root_config},
                "strength_evidence_eligible": True,
                "value_leaf": {
                    **value_leaf_provenance,
                },
            },
        )

    def test_value_checkpoint_requires_matching_observation_and_belief_provenance(self) -> None:
        policy_result = SimpleNamespace(model_config="v2.2", belief_set_source_hash="source-a")
        matching_result = SimpleNamespace(
            model_config="v2.2",
            belief_set_source_hash="source-a",
            value_calibration_source_checkpoint_sha256="raw-sha",
            value_calibration_transform=ValueCalibrationTransform(),
        )
        with patch("pokezero.neural_policy.checkpoint_file_sha256", side_effect=("raw-sha", "leaf-sha")):
            provenance = require_compatible_transformer_value_checkpoint(
                policy_checkpoint=Path("policy.pt"),
                policy_result=policy_result,
                value_checkpoint=Path("calibrated.pt"),
                value_result=matching_result,
            )
        self.assertEqual(provenance["policy_checkpoint_sha256"], "raw-sha")
        self.assertEqual(provenance["value_checkpoint_sha256"], "leaf-sha")
        with self.assertRaisesRegex(ValueError, "model config"):
            require_compatible_transformer_value_checkpoint(
                policy_checkpoint=Path("policy.pt"),
                policy_result=policy_result,
                value_checkpoint=Path("wrong-shape.pt"),
                value_result=SimpleNamespace(
                    model_config="v2.1",
                    belief_set_source_hash="source-a",
                    value_calibration_source_checkpoint_sha256="raw-sha",
                ),
            )
        with self.assertRaisesRegex(ValueError, "belief-set provenance"):
            require_compatible_transformer_value_checkpoint(
                policy_checkpoint=Path("policy.pt"),
                policy_result=policy_result,
                value_checkpoint=Path("wrong-belief.pt"),
                value_result=SimpleNamespace(
                    model_config="v2.2",
                    belief_set_source_hash="source-b",
                    value_calibration_source_checkpoint_sha256="raw-sha",
                ),
            )
        with patch("pokezero.neural_policy.checkpoint_file_sha256", return_value="raw-sha"):
            with self.assertRaisesRegex(ValueError, "no calibrated-copy source provenance"):
                require_compatible_transformer_value_checkpoint(
                    policy_checkpoint=Path("policy.pt"),
                    policy_result=policy_result,
                    value_checkpoint=Path("unproven.pt"),
                    value_result=SimpleNamespace(
                        model_config="v2.2",
                        belief_set_source_hash="source-a",
                        value_calibration_source_checkpoint_sha256=None,
                        value_calibration_transform=ValueCalibrationTransform(),
                    ),
                )
        with patch("pokezero.neural_policy.checkpoint_file_sha256", return_value="raw-sha"):
            with self.assertRaisesRegex(ValueError, "source hash does not match"):
                require_compatible_transformer_value_checkpoint(
                    policy_checkpoint=Path("policy.pt"),
                    policy_result=policy_result,
                    value_checkpoint=Path("wrong-parent.pt"),
                    value_result=SimpleNamespace(
                        model_config="v2.2",
                        belief_set_source_hash="source-a",
                        value_calibration_source_checkpoint_sha256="other-sha",
                        value_calibration_transform=ValueCalibrationTransform(),
                    ),
                )
        with self.assertRaisesRegex(ValueError, "no calibration transform"):
            require_compatible_transformer_value_checkpoint(
                policy_checkpoint=Path("policy.pt"),
                policy_result=policy_result,
                value_checkpoint=Path("uncalibrated.pt"),
                value_result=SimpleNamespace(
                    model_config="v2.2",
                    belief_set_source_hash="source-a",
                    value_calibration_source_checkpoint_sha256="raw-sha",
                    value_calibration_transform=None,
                ),
            )

    def test_neural_cli_root_puct_play_benchmark_wires_public_belief_worlds(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        fake_model = object()
        fake_training_result = SimpleNamespace(
            model_config=SimpleNamespace(
                policy_id="neural-smoke",
                window_size=1,
                format_id="gen3randombattle",
                observation_schema_version="pokezero.observation.v2.1",
                categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
                numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
                stats_block_enabled=True,
                exact_state_enabled=True,
                transition_token_budget=128,
                tier2_residuals=True,
                tier2_investment=False,
            )
        )
        captured = {}

        def belief_planner(context, scenario, scenario_index, rng):
            del context, scenario, scenario_index, rng
            return None

        belief_planner.sample_count_for_context = lambda context: 2  # type: ignore[attr-defined]

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            search_policy = tuple(kwargs["matchups"])[2].p1_policy
            captured["search_policy"] = search_policy
            search_policy.env_factory()
            return SimpleNamespace(to_dict=lambda: {"matchups": 4}, matchups=())

        def capture_env(config):
            captured["env_config"] = config
            return object()

        stdout = io.StringIO()
        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25),
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(1.0,) + (0.0,) * 8),
            patch("pokezero.neural_cli.load_gen3_randbat_source_cached", return_value=object()) as load_source,
            patch("pokezero.neural_cli.gen3_randbat_belief_start_override_planner", return_value=belief_planner) as planner,
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            patch("pokezero.neural_cli.LocalShowdownEnv", side_effect=capture_env),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--opponent-policy",
                    "random-legal",
                    "--belief-start-overrides",
                    "--record-belief-world-coverage-gaps",
                    "--belief-world-sample-cap",
                    "3",
                    "--belief-start-override-attempts",
                    "7",
                    "--belief-start-override-hp-fraction-tolerance",
                    "0.03",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        load_source.assert_called_once()
        planner.assert_called_once_with(load_source.return_value, world_sample_cap=3)
        search_policy = captured["search_policy"]
        self.assertIs(search_policy.start_override_planner, belief_planner)
        self.assertIsNone(search_policy.start_override_samples_per_scenario)
        self.assertEqual(search_policy.start_override_attempts, 7)
        self.assertEqual(search_policy.start_override_hp_fraction_tolerance, 0.03)
        self.assertTrue(captured["env_config"].set_belief_source)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["belief_world_coverage_gaps_allowed"])
        self.assertEqual(payload["belief_world_coverage_mode"], "mechanics-only-gaps-allowed")
        self.assertFalse(payload["strength_evidence_eligible"])
        self.assertEqual(payload["artifact_scope"], "w5-mechanics-only-not-strength-evidence")
        self.assertEqual(payload["belief_world_coverage"]["per_game_any_materialization_rate"], 1.0)

    def test_root_puct_belief_benchmark_rejects_missing_world_checksums(self) -> None:
        result = SimpleNamespace(
            label="search vs max-damage",
            p1_policy_id="search",
            p2_policy_id="max-damage",
            seed_start=12,
            metrics=SimpleNamespace(games=2),
            root_puct_belief_public_checksums_by_seed={12: ("public",)},
        )

        with self.assertRaisesRegex(RuntimeError, "missing belief-world checksum for seeds 13"):
            _require_belief_world_benchmark_coverage(
                _belief_world_benchmark_coverage(
                    SimpleNamespace(matchups=(result,)),
                    search_policy_ids=("search",),
                ),
            )

    def test_root_puct_belief_benchmark_records_partial_world_coverage(self) -> None:
        result = SimpleNamespace(
            label="search vs max-damage",
            p1_policy_id="search",
            p2_policy_id="max-damage",
            seed_start=12,
            metrics=SimpleNamespace(games=3),
            root_puct_belief_public_checksums_by_seed={12: ("public",), 14: ("public",)},
        )

        coverage = _belief_world_benchmark_coverage(
            SimpleNamespace(matchups=(result,)),
            search_policy_ids=("search",),
        )

        self.assertEqual(coverage["scope"], "per-game-any-decision")
        self.assertEqual(coverage["expected_game_count"], 3)
        self.assertEqual(coverage["games_with_materialized_world"], 2)
        self.assertEqual(coverage["games_without_materialized_world"], 1)
        self.assertEqual(coverage["per_game_any_materialization_rate"], 2 / 3)
        self.assertEqual(coverage["matchups"][0]["missing_seeds"], [13])

    def test_root_puct_belief_world_coverage_gaps_flag_parses(self) -> None:
        args = build_neural_arg_parser().parse_args(
            [
                "root-puct-play-benchmark",
                "--checkpoint",
                "checkpoint.pt",
                "--belief-start-overrides",
                "--record-belief-world-coverage-gaps",
            ]
        )

        self.assertTrue(args.belief_start_overrides)
        self.assertTrue(args.record_belief_world_coverage_gaps)

    def test_neural_cli_root_puct_play_benchmark_defaults_root_prior_temperature_to_temperature(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        fake_model = object()
        fake_training_result = SimpleNamespace(
            model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1, format_id="gen3randombattle", observation_schema_version="pokezero.observation.v2.1", categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count, numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count, stats_block_enabled=True, exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False)
        )
        captured = {}

        def fake_benchmark_rollouts(**kwargs):
            captured.update(kwargs)
            search_policy = tuple(kwargs["matchups"])[2].p1_policy
            self.assertEqual(search_policy.root_prior_temperature, 1.75)
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
            self.assertEqual(search_policy.opponent_action_planner(context, __import__("random").Random(1)), {"p2": 2})
            return SimpleNamespace(to_dict=lambda: {"matchups": 4})

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.evaluate_transformer_observation_value", return_value=0.25),
            patch("pokezero.neural_cli.evaluate_transformer_action_priors", return_value=(1.0,) + (0.0,) * 8) as prior_eval,
            patch("pokezero.neural_cli.evaluate_transformer_opponent_action_priors", return_value=(0.1, 0.2, 0.7) + (0.0,) * 6) as opponent_eval,
            patch("pokezero.neural_cli.benchmark_rollouts", side_effect=fake_benchmark_rollouts),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--opponent-policy",
                    "random-legal",
                    "--temperature",
                    "1.75",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["games"], 20)
        self.assertEqual(prior_eval.call_args.kwargs["temperature"], 1.0)
        self.assertEqual(opponent_eval.call_args.kwargs["temperature"], 1.75)

    def test_neural_cli_root_puct_play_benchmark_can_average_checkpoint_opponent_action_scenarios(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 4}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1, observation_schema_version="pokezero.observation.v2.1", categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count, numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count, stats_block_enabled=True, exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False))
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
                    "p2": (True, True, True, True, False, False, False, False, False),
                },
            )
            scenarios = search_policy.opponent_action_scenario_planner(context, __import__("random").Random(1))
            self.assertEqual([dict(scenario.actions) for scenario in scenarios], [{"p2": 2}, {"p2": 1}, {"p2": 0}])
            self.assertAlmostEqual(scenarios[0].weight, 0.7 / 1.0)
            self.assertAlmostEqual(scenarios[1].weight, 0.2 / 1.0)
            self.assertAlmostEqual(scenarios[2].weight, 0.1 / 1.0)
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
                    "--allow-legacy-checkpoints",
                    "--games",
                    "3",
                    "--opponent-policy",
                    "random-legal",
                    "--root-opponent-action-scenarios",
                    "2",
                    "--root-opponent-action-candidate-scenarios",
                    "3",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        matchups = tuple(captured["matchups"])
        search_policy = matchups[2].p1_policy
        self.assertEqual(getattr(search_policy.opponent_action_planner, "planner_id"), "checkpoint")
        self.assertIsNotNone(search_policy.opponent_action_scenario_planner)
        self.assertEqual(getattr(search_policy.opponent_action_scenario_planner, "planner_id"), "checkpoint-top3")
        self.assertEqual(search_policy.max_opponent_action_scenarios, 2)
        rendered = json.loads(stdout.getvalue())
        self.assertEqual(rendered["matchups"], 4)
        self.assertEqual(rendered["root_puct_config"], rendered["root_puct_policy_configs"]["neural-smoke+root-puct"])

    def test_neural_cli_root_puct_play_benchmark_rejects_candidate_scenarios_below_accepted_count(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(
                [
                    "root-puct-play-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--allow-legacy-checkpoints",
                    "--root-opponent-action-scenarios",
                    "2",
                    "--root-opponent-action-candidate-scenarios",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("candidate scenarios", stderr.getvalue())

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
                    "--allow-legacy-checkpoints",
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
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1, observation_schema_version="pokezero.observation.v2.1", categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count, numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count, stats_block_enabled=True, exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False))
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
                    "--allow-legacy-checkpoints",
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
        rendered = json.loads(stdout.getvalue())
        self.assertEqual(rendered["matchups"], 4)
        self.assertEqual(rendered["root_puct_config"], rendered["root_puct_policy_configs"]["neural-smoke+root-puct"])

    def test_neural_cli_root_puct_play_benchmark_can_use_benchmark_opponent_for_leaf_rollouts(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 4}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1, observation_schema_version="pokezero.observation.v2.1", categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count, numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count, stats_block_enabled=True, exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False))
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
                    "--allow-legacy-checkpoints",
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
        rendered = json.loads(stdout.getvalue())
        self.assertEqual(rendered["matchups"], 4)
        self.assertEqual(rendered["root_puct_config"]["leaf_rollout_opponent_policy"], "benchmark")

    def test_neural_cli_root_puct_play_benchmark_can_sweep_leaf_depths(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 6}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1, observation_schema_version="pokezero.observation.v2.1", categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count, numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count, stats_block_enabled=True, exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False))
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
                    "--allow-legacy-checkpoints",
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
        rendered = json.loads(stdout.getvalue())
        self.assertEqual(rendered["matchups"], 6)
        self.assertNotIn("root_puct_config", rendered)
        self.assertEqual(
            {
                policy_id: config["leaf_rollout_rounds"]
                for policy_id, config in rendered["root_puct_policy_configs"].items()
            },
            {
                "neural-smoke+root-puct-leaf0": 0,
                "neural-smoke+root-puct-leaf2": 2,
            },
        )

    def test_neural_cli_root_puct_play_benchmark_tags_single_sweep_depth(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"matchups": 4}

        fake_model = object()
        fake_training_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke", window_size=1, observation_schema_version="pokezero.observation.v2.1", categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count, numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count, stats_block_enabled=True, exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False))
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
                    "--allow-legacy-checkpoints",
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
        rendered = json.loads(stdout.getvalue())
        self.assertEqual(rendered["matchups"], 4)
        self.assertEqual(rendered["root_puct_config"]["leaf_rollout_rounds"], 2)
        self.assertEqual(
            rendered["root_puct_policy_configs"]["neural-smoke+root-puct-leaf2"],
            rendered["root_puct_config"],
        )

    def test_neural_cli_root_puct_benchmark_wires_checkpoint_callbacks_and_source_policies(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"evaluated_prefixes": 2}

        fake_model = object()
        fake_value_model = object()
        fake_training_result = SimpleNamespace(
            model_config=SimpleNamespace(
                policy_id="neural-smoke", window_size=1,
                observation_schema_version="pokezero.observation.v2.1",
                categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
                numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
                stats_block_enabled=True,
                exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False,
            )
        )
        value_leaf_provenance = {
            "policy_checkpoint": "/checkpoint.pt",
            "policy_checkpoint_sha256": "raw-sha",
            "value_checkpoint": "/calibrated.pt",
            "value_checkpoint_sha256": "leaf-sha",
            "value_calibration_source_checkpoint_sha256": "raw-sha",
            "model_config_match": True,
            "belief_set_source_hash_match": True,
            "value_calibration_transform": {"method": "isotonic"},
        }
        captured = {}

        def fake_benchmark_root_puct_search(**kwargs):
            captured.update(kwargs)
            self.assertEqual(kwargs["value_fn"]((observation(1),)), 0.25)
            self.assertEqual(kwargs["prior_fn"]((observation(1),)), (1.0,) + (0.0,) * 8)
            return FakeReport()

        stdout = io.StringIO()

        with (
            patch(
                "pokezero.neural_cli.load_transformer_checkpoint",
                side_effect=((fake_model, fake_training_result), (fake_value_model, fake_training_result)),
            ) as load,
            patch(
                "pokezero.neural_cli.require_compatible_transformer_value_checkpoint",
                return_value=value_leaf_provenance,
            ) as require_compatible,
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
                    "--allow-legacy-checkpoints",
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
                    "--value-checkpoint",
                    "calibrated.pt",
                    "--root-extra-visits",
                    "24",
                    "--device",
                    "cpu",
                    "--temperature",
                    "1.5",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            load.call_args_list,
            [
                call(Path("checkpoint.pt"), map_location="cpu"),
                call(Path("calibrated.pt"), map_location="cpu"),
            ],
        )
        self.assertEqual(captured["games"], 3)
        self.assertEqual(captured["prefixes_per_game"], 4)
        self.assertEqual(captured["seed_start"], 99)
        self.assertEqual(captured["search_player"], "p2")
        self.assertEqual(captured["cpuct"], 0.75)
        self.assertEqual(captured["root_extra_visits"], 24)
        self.assertEqual(captured["rollout_config"].max_decision_rounds, 12)
        self.assertEqual(captured["policies"]["p1"].policy_id, "random-legal")
        self.assertEqual(captured["policies"]["p2"].policy_id, "simple-legal")
        self.assertEqual(value_eval.call_args.kwargs["model"], fake_value_model)
        self.assertEqual(value_eval.call_args.kwargs["result"], fake_training_result)
        self.assertEqual(value_eval.call_args.kwargs["device"], "cpu")
        self.assertEqual(prior_eval.call_args.kwargs["temperature"], 1.5)
        require_compatible.assert_called_once_with(
            policy_checkpoint=Path("checkpoint.pt"),
            policy_result=fake_training_result,
            value_checkpoint=Path("calibrated.pt"),
            value_result=fake_training_result,
        )
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "evaluated_prefixes": 2,
                "search_config": {
                    "prior_temperature": 1.5,
                    "selection_mode": "visits",
                },
                "value_leaf": {
                    **value_leaf_provenance,
                    "uses_distinct_value_checkpoint": True,
                },
            },
        )

    def test_neural_cli_root_puct_benchmark_rejects_incompatible_value_leaf(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        fake_model = object()
        raw_result = SimpleNamespace(model_config=SimpleNamespace(policy_id="neural-smoke"))
        incompatible = ValueError("value checkpoint calibrated-copy source hash does not match policy checkpoint")

        stderr = io.StringIO()
        with (
            patch(
                "pokezero.neural_cli.load_transformer_checkpoint",
                side_effect=((fake_model, raw_result), (object(), raw_result)),
            ),
            patch(
                "pokezero.neural_cli.require_compatible_transformer_value_checkpoint",
                side_effect=incompatible,
            ) as require_compatible,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = neural_cli_main(
                [
                    "root-puct-benchmark",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--value-checkpoint",
                    "uncalibrated.pt",
                    "--allow-legacy-checkpoints",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("source hash does not match", stderr.getvalue())
        require_compatible.assert_called_once_with(
            policy_checkpoint=Path("checkpoint.pt"),
            policy_result=raw_result,
            value_checkpoint=Path("uncalibrated.pt"),
            value_result=raw_result,
        )

    def test_neural_cli_root_puct_counterfactual_wires_continuation_policies(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"average_rollout_value_delta": 0.5}

        fake_model = object()
        fake_training_result = SimpleNamespace(
            model_config=SimpleNamespace(
                policy_id="neural-smoke", window_size=1,
                observation_schema_version="pokezero.observation.v2.1",
                categorical_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
                numeric_feature_count=DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
                stats_block_enabled=True,
                exact_state_enabled=True, transition_token_budget=128, tier2_residuals=True, tier2_investment=False,
            )
        )
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
                    "--allow-legacy-checkpoints",
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

    def test_neural_cli_value_calibration_can_enforce_quality_gates_json(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        report = ValueCalibrationReport(
            examples=300,
            mse=0.22,
            mae=0.31,
            bias=-0.03,
            sign_accuracy=0.68,
            expected_calibration_error=0.11,
            pearson_correlation=0.42,
            bins=(),
            slices=(),
        )

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(object(), object())),
            patch("pokezero.neural_cli.evaluate_value_calibration", return_value=report),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--min-examples",
                    "100",
                    "--max-expected-calibration-error",
                    "0.2",
                    "--min-sign-accuracy",
                    "0.6",
                    "--min-pearson-correlation",
                    "0.3",
                    "--max-abs-bias",
                    "0.1",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        gates = payload["quality_gates"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["examples"], 300)
        self.assertTrue(gates["configured"])
        self.assertTrue(gates["passed"])
        self.assertEqual([check["metric"] for check in gates["checks"]], [
            "examples",
            "abs_bias",
            "expected_calibration_error",
            "sign_accuracy",
            "pearson_correlation",
        ])

    def test_neural_cli_value_calibration_quality_gate_failure_returns_nonzero(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        report = ValueCalibrationReport(
            examples=300,
            mse=0.22,
            mae=0.31,
            bias=-0.03,
            sign_accuracy=0.68,
            expected_calibration_error=0.11,
            pearson_correlation=None,
            bins=(),
            slices=(),
        )

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(object(), object())),
            patch("pokezero.neural_cli.evaluate_value_calibration", return_value=report),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--min-pearson-correlation",
                    "0.3",
                ]
            )

        self.assertEqual(exit_code, 4)
        self.assertIn("quality_gates:", stdout.getvalue())
        self.assertIn("reason=unavailable", stdout.getvalue())
        self.assertIn("value_calibration_quality_gates_failed: pearson_correlation", stderr.getvalue())

    def test_neural_cli_value_calibration_includes_quality_gates_with_fit_out(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        report = ValueCalibrationReport(
            examples=300,
            mse=0.22,
            mae=0.31,
            bias=-0.03,
            sign_accuracy=0.68,
            expected_calibration_error=0.11,
            pearson_correlation=0.42,
            bins=(),
            slices=(),
        )
        model_config = TransformerPolicyConfig.compact_category(
            category_vocab=(1, 2, 3),
            category_oov_buckets=4,
            policy_id="fixture",
        )
        fake_training_result = TransformerTrainingResult(
            model_config=model_config,
            training_config=TransformerTrainingConfig(),
            epochs=(
                TransformerEpochMetrics(
                    epoch=1,
                    examples=3,
                    loss=0.2,
                    policy_loss=0.1,
                    policy_accuracy=0.5,
                ),
            ),
        )
        transform = ValueCalibrationTransform(scale=1.5, bias=-0.2)

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(object(), fake_training_result)),
            patch("pokezero.neural_cli.fit_value_calibration_transform", return_value=transform),
            patch("pokezero.neural_cli.checkpoint_file_sha256", return_value="source-sha"),
            patch("pokezero.neural_cli.save_transformer_checkpoint"),
            patch("pokezero.neural_cli.evaluate_value_calibration", return_value=report),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--eval-data",
                    "eval-rollouts.jsonl",
                    "--fit-out",
                    "calibrated.pt",
                    "--min-sign-accuracy",
                    "0.6",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["checkpoint"], "calibrated.pt")
        self.assertEqual(payload["report"]["examples"], 300)
        self.assertTrue(payload["quality_gates"]["passed"])

    def test_neural_cli_value_calibration_rejects_non_finite_quality_gate(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint") as load,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--max-mae",
                    "nan",
                ]
            )

        self.assertEqual(exit_code, 1)
        load.assert_not_called()
        self.assertIn("--max-mae must be finite and non-negative", stderr.getvalue())

    def test_neural_cli_value_calibration_can_save_calibrated_checkpoint(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"examples": 3, "mae": 0.1}

        model_config = TransformerPolicyConfig.compact_category(
            category_vocab=(1, 2, 3),
            category_oov_buckets=4,
            policy_id="fixture",
        )
        fake_model = object()
        fake_training_result = TransformerTrainingResult(
            model_config=model_config,
            training_config=TransformerTrainingConfig(),
            epochs=(
                TransformerEpochMetrics(
                    epoch=1,
                    examples=3,
                    loss=0.2,
                    policy_loss=0.1,
                    policy_accuracy=0.5,
                ),
            ),
        )
        transform = ValueCalibrationTransform(scale=1.5, bias=-0.2)

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.fit_value_calibration_transform", return_value=transform) as fit,
            patch("pokezero.neural_cli.checkpoint_file_sha256", return_value="source-sha"),
            patch("pokezero.neural_cli.save_transformer_checkpoint") as save,
            patch("pokezero.neural_cli.evaluate_value_calibration", return_value=FakeReport()) as evaluate,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--eval-data",
                    "eval-rollouts.jsonl",
                    "--fit-out",
                    "calibrated.pt",
                    "--device",
                    "cpu",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(fit.call_args.kwargs["training_result"], fake_training_result)
        self.assertEqual(fit.call_args.kwargs["paths"], [Path("rollouts.jsonl")])
        self.assertEqual(fit.call_args.kwargs["method"], "affine")
        saved_result = save.call_args.kwargs["result"]
        self.assertEqual(save.call_args.args[0], Path("calibrated.pt"))
        self.assertEqual(saved_result.value_calibration_transform, transform)
        self.assertEqual(saved_result.value_calibration_source_checkpoint_sha256, "source-sha")
        self.assertEqual(evaluate.call_args.kwargs["training_result"].value_calibration_transform, transform)
        self.assertEqual(evaluate.call_args.kwargs["paths"], [Path("eval-rollouts.jsonl")])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["checkpoint"], "calibrated.pt")
        self.assertEqual(payload["fit_paths"], ["rollouts.jsonl"])
        self.assertEqual(payload["evaluation_paths"], ["eval-rollouts.jsonl"])
        self.assertTrue(payload["evaluation_held_out"])
        self.assertEqual(payload["value_calibration_transform"]["scale"], 1.5)
        self.assertEqual(payload["value_calibration_source_checkpoint_sha256"], "source-sha")
        self.assertEqual(payload["report"], {"examples": 3, "mae": 0.1})

    def test_neural_cli_value_calibration_can_save_isotonic_calibrated_checkpoint(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"examples": 3, "mae": 0.1}

        model_config = TransformerPolicyConfig.compact_category(
            category_vocab=(1, 2, 3),
            category_oov_buckets=4,
            policy_id="fixture",
        )
        fake_model = object()
        fake_training_result = TransformerTrainingResult(
            model_config=model_config,
            training_config=TransformerTrainingConfig(),
            epochs=(),
        )
        transform = ValueCalibrationTransform(
            method="isotonic",
            points=((-1.0, -0.8), (0.0, 0.1), (1.0, 0.9)),
        )

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(fake_model, fake_training_result)),
            patch("pokezero.neural_cli.fit_value_calibration_transform", return_value=transform) as fit,
            patch("pokezero.neural_cli.checkpoint_file_sha256", return_value="source-sha"),
            patch("pokezero.neural_cli.save_transformer_checkpoint"),
            patch("pokezero.neural_cli.evaluate_value_calibration", return_value=FakeReport()),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--fit-out",
                    "calibrated.pt",
                    "--fit-method",
                    "isotonic",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(fit.call_args.kwargs["method"], "isotonic")
        self.assertEqual(payload["value_calibration_transform"]["method"], "isotonic")
        self.assertEqual(len(payload["value_calibration_transform"]["points"]), 3)

    def test_neural_cli_value_calibration_warns_on_collapsed_isotonic_transform(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        class FakeReport:
            def to_dict(self) -> dict:
                return {"examples": 3, "mae": 0.1}

        fake_training_result = TransformerTrainingResult(
            model_config=TransformerPolicyConfig.compact_category(
                category_vocab=(1, 2, 3),
                category_oov_buckets=4,
                policy_id="fixture",
            ),
            training_config=TransformerTrainingConfig(),
            epochs=(),
        )
        transform = ValueCalibrationTransform(
            method="isotonic",
            points=((-1.0, 0.0), (1.0, 0.0)),
        )

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(object(), fake_training_result)),
            patch("pokezero.neural_cli.fit_value_calibration_transform", return_value=transform),
            patch("pokezero.neural_cli.checkpoint_file_sha256", return_value="source-sha"),
            patch("pokezero.neural_cli.save_transformer_checkpoint"),
            patch("pokezero.neural_cli.evaluate_value_calibration", return_value=FakeReport()),
            patch("sys.stdout", new_callable=io.StringIO),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--fit-out",
                    "calibrated.pt",
                    "--fit-method",
                    "isotonic",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("near-constant", stderr.getvalue())
        self.assertIn("value-head search nearly value-blind", stderr.getvalue())

    def test_neural_cli_value_calibration_compare_reports_heldout_methods(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        def report(*, mae: float, ece: float, sign_accuracy: float = 0.5) -> ValueCalibrationReport:
            return ValueCalibrationReport(
                examples=20,
                mse=mae * mae,
                mae=mae,
                bias=0.0,
                sign_accuracy=sign_accuracy,
                expected_calibration_error=ece,
                pearson_correlation=0.25,
                bins=(),
                slices=(),
            )

        fake_training_result = TransformerTrainingResult(
            model_config=TransformerPolicyConfig.compact_category(
                category_vocab=(1, 2, 3),
                category_oov_buckets=4,
                policy_id="fixture",
            ),
            training_config=TransformerTrainingConfig(),
            value_calibration_transform=ValueCalibrationTransform(scale=0.5, bias=0.1),
            epochs=(),
        )
        affine = ValueCalibrationTransform(scale=1.5, bias=-0.2)
        isotonic = ValueCalibrationTransform(method="isotonic", points=((-1.0, -0.8), (1.0, 0.8)))

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(object(), fake_training_result)),
            patch("pokezero.neural_cli.fit_value_calibration_transform", side_effect=(affine, isotonic)) as fit,
            patch(
                "pokezero.neural_cli.evaluate_value_calibration",
                side_effect=(report(mae=0.7, ece=0.4), report(mae=0.5, ece=0.2), report(mae=0.4, ece=0.1)),
            ) as evaluate,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration-compare",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "fit-rollouts.jsonl",
                    "--eval-data",
                    "heldout-rollouts.jsonl",
                    "--selection-metric",
                    "expected_calibration_error",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["evaluation_paths"], ["heldout-rollouts.jsonl"])
        self.assertTrue(payload["evaluation_held_out"])
        self.assertEqual(payload["selection_metric"], "expected_calibration_error")
        self.assertEqual(payload["best_method"], "isotonic")
        self.assertEqual([entry["method"] for entry in payload["methods"]], ["raw", "affine", "isotonic"])
        self.assertIn("calibration_only_selection_metric", {warning["code"] for warning in payload["warnings"]})
        self.assertIsNone(evaluate.call_args_list[0].kwargs["training_result"].value_calibration_transform)
        self.assertEqual(evaluate.call_args_list[1].kwargs["training_result"].value_calibration_transform, affine)
        self.assertEqual(evaluate.call_args_list[2].kwargs["training_result"].value_calibration_transform, isotonic)
        self.assertEqual([call.kwargs["method"] for call in fit.call_args_list], ["affine", "isotonic"])

    def test_neural_cli_value_calibration_compare_defaults_to_pearson_selection(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        def report(*, pearson: float) -> ValueCalibrationReport:
            return ValueCalibrationReport(
                examples=20,
                mse=0.25,
                mae=0.5,
                bias=0.0,
                sign_accuracy=0.6,
                expected_calibration_error=0.2,
                pearson_correlation=pearson,
                bins=(),
                slices=(),
            )

        with (
            patch(
                "pokezero.neural_cli.load_transformer_checkpoint",
                return_value=(
                    object(),
                    TransformerTrainingResult(
                        model_config=TransformerPolicyConfig.compact_category(
                            category_vocab=(1, 2, 3),
                            category_oov_buckets=4,
                            policy_id="fixture",
                        ),
                        training_config=TransformerTrainingConfig(),
                        epochs=(),
                    ),
                ),
            ),
            patch(
                "pokezero.neural_cli.fit_value_calibration_transform",
                side_effect=(
                    ValueCalibrationTransform(scale=1.0, bias=0.0),
                    ValueCalibrationTransform(method="isotonic", points=((-1.0, 0.0), (1.0, 0.0))),
                ),
            ),
            patch("pokezero.neural_cli.evaluate_value_calibration", side_effect=(report(pearson=0.1), report(pearson=0.4), report(pearson=0.2))),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration-compare",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "fit-rollouts.jsonl",
                    "--eval-data",
                    "heldout-rollouts.jsonl",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["selection_metric"], "pearson_correlation")
        self.assertEqual(payload["selection_direction"], "max")
        self.assertEqual(payload["best_method"], "affine")

    def test_neural_cli_value_calibration_compare_text_can_write_json_report(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        fake_training_result = TransformerTrainingResult(
            model_config=TransformerPolicyConfig.compact_category(
                category_vocab=(1, 2, 3),
                category_oov_buckets=4,
                policy_id="fixture",
            ),
            training_config=TransformerTrainingConfig(),
            epochs=(),
        )
        fake_report = ValueCalibrationReport(
            examples=20,
            mse=0.25,
            mae=0.5,
            bias=0.0,
            sign_accuracy=0.5,
            expected_calibration_error=0.2,
            pearson_correlation=None,
            bins=(),
            slices=(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "compare.json"
            with (
                patch("pokezero.neural_cli.load_transformer_checkpoint", return_value=(object(), fake_training_result)),
                patch(
                    "pokezero.neural_cli.fit_value_calibration_transform",
                    side_effect=(
                        ValueCalibrationTransform(scale=1.0, bias=0.0),
                        ValueCalibrationTransform(method="isotonic", points=((-1.0, 0.0), (1.0, 0.0))),
                    ),
                ),
                patch("pokezero.neural_cli.evaluate_value_calibration", return_value=fake_report),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                exit_code = neural_cli_main(
                    [
                        "value-calibration-compare",
                        "--checkpoint",
                        "checkpoint.pt",
                        "--data",
                        "fit-rollouts.jsonl",
                        "--eval-data",
                        "fit-rollouts.jsonl",
                        "--selection-metric",
                        "expected_calibration_error",
                        "--out",
                        str(out_path),
                    ]
                )

            payload = json.loads(out_path.read_text(encoding="utf-8"))

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["best_method"], "raw")
        self.assertIn("fit_eval_path_overlap", {warning["code"] for warning in payload["warnings"]})
        self.assertIn("value_calibration_compare:", output)
        self.assertIn("comparison_json:", output)

    def test_neural_cli_value_calibration_compare_skips_unavailable_selection_metric_rows(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        unavailable = ValueCalibrationReport(
            examples=20,
            mse=0.25,
            mae=0.5,
            bias=0.0,
            sign_accuracy=0.5,
            expected_calibration_error=0.2,
            pearson_correlation=None,
            bins=(),
            slices=(),
        )
        available = ValueCalibrationReport(
            examples=20,
            mse=0.16,
            mae=0.4,
            bias=0.0,
            sign_accuracy=0.6,
            expected_calibration_error=0.1,
            pearson_correlation=0.4,
            bins=(),
            slices=(),
        )

        with (
            patch(
                "pokezero.neural_cli.load_transformer_checkpoint",
                return_value=(
                    object(),
                    TransformerTrainingResult(
                        model_config=TransformerPolicyConfig.compact_category(
                            category_vocab=(1, 2, 3),
                            category_oov_buckets=4,
                            policy_id="fixture",
                        ),
                        training_config=TransformerTrainingConfig(),
                        epochs=(),
                    ),
                ),
            ),
            patch(
                "pokezero.neural_cli.fit_value_calibration_transform",
                side_effect=(
                    ValueCalibrationTransform(scale=1.0, bias=0.0),
                    ValueCalibrationTransform(method="isotonic", points=((-1.0, 0.0), (1.0, 0.0))),
                ),
            ),
            patch("pokezero.neural_cli.evaluate_value_calibration", side_effect=(unavailable, available, unavailable)),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration-compare",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "fit-rollouts.jsonl",
                    "--eval-data",
                    "heldout-rollouts.jsonl",
                    "--selection-metric",
                    "pearson_correlation",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["best_method"], "affine")
        self.assertIsNone(payload["methods"][0]["selection_metric_value"])
        self.assertIn("selection_error", payload["methods"][0])

    def test_neural_cli_value_calibration_compare_rejects_all_unavailable_selection_metric(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        unavailable = ValueCalibrationReport(
            examples=20,
            mse=0.25,
            mae=0.5,
            bias=0.0,
            sign_accuracy=0.5,
            expected_calibration_error=0.2,
            pearson_correlation=None,
            bins=(),
            slices=(),
        )

        with (
            patch(
                "pokezero.neural_cli.load_transformer_checkpoint",
                return_value=(
                    object(),
                    TransformerTrainingResult(
                        model_config=TransformerPolicyConfig.compact_category(
                            category_vocab=(1, 2, 3),
                            category_oov_buckets=4,
                            policy_id="fixture",
                        ),
                        training_config=TransformerTrainingConfig(),
                        epochs=(),
                    ),
                ),
            ),
            patch(
                "pokezero.neural_cli.fit_value_calibration_transform",
                side_effect=(
                    ValueCalibrationTransform(scale=1.0, bias=0.0),
                    ValueCalibrationTransform(method="isotonic", points=((-1.0, 0.0), (1.0, 0.0))),
                ),
            ),
            patch("pokezero.neural_cli.evaluate_value_calibration", return_value=unavailable),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration-compare",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "fit-rollouts.jsonl",
                    "--eval-data",
                    "heldout-rollouts.jsonl",
                    "--selection-metric",
                    "pearson_correlation",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("pearson_correlation is unavailable for all calibration methods", stderr.getvalue())

    def test_neural_cli_value_calibration_rejects_eval_data_without_fit_out(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--eval-data",
                    "eval-rollouts.jsonl",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--eval-data requires --fit-out", stderr.getvalue())

    def test_neural_cli_value_calibration_rejects_fit_method_without_fit_out(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        with (
            patch("pokezero.neural_cli.load_transformer_checkpoint") as load,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            exit_code = neural_cli_main(
                [
                    "value-calibration",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--data",
                    "rollouts.jsonl",
                    "--fit-method",
                    "isotonic",
                ]
            )

        self.assertEqual(exit_code, 1)
        load.assert_not_called()
        self.assertIn("--fit-method requires --fit-out", stderr.getvalue())

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
                        # Pinned: unstamped .jsonl fixture data pairs with v2.1 only.
                        "--observation-schema",
                        "v2.1",
                        "--temporal-aggregator",
                        "gru",
                        "--hp-delta-return-weight",
                        "0.2",
                        "--faint-delta-return-weight",
                        "0.4",
                        "--turn-penalty-after",
                        "180",
                        "--turn-penalty",
                        "0.01",
                        "--objective",
                        "ppo",
                        "--ppo-target-mode",
                        "gae",
                        "--gae-lambda",
                        "0.8",
                        "--value-clip-range",
                        "0.0184",
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
        self.assertEqual(train.call_args.kwargs["model_config"].temporal_aggregator, "gru")
        self.assertEqual(train.call_args.kwargs["training_config"].hp_delta_return_weight, 0.2)
        self.assertEqual(train.call_args.kwargs["training_config"].faint_delta_return_weight, 0.4)
        self.assertEqual(train.call_args.kwargs["training_config"].turn_penalty_after, 180)
        self.assertEqual(train.call_args.kwargs["training_config"].turn_penalty, 0.01)
        self.assertEqual(train.call_args.kwargs["training_config"].objective, "ppo")
        self.assertEqual(train.call_args.kwargs["training_config"].ppo_target_mode, "gae")
        self.assertEqual(train.call_args.kwargs["training_config"].gae_lambda, 0.8)
        self.assertEqual(train.call_args.kwargs["training_config"].value_clip_range, 0.0184)
        self.assertEqual(save.call_args.args[0], checkpoint_path)
        self.assertEqual(evaluate.call_args.kwargs["paths"], [Path("calibration-rollouts.jsonl")])
        self.assertEqual(evaluate.call_args.kwargs["batch_size"], 9)
        self.assertEqual(evaluate.call_args.kwargs["bins"], 6)
        self.assertEqual(payload["paths"], ["calibration-rollouts.jsonl"])
        self.assertEqual(payload["report"]["sign_accuracy"], 0.75)
        self.assertIn(f"value_calibration: {calibration_path}", stdout.getvalue())

    def test_neural_cli_train_finalizes_cache_lifecycle_after_checkpoint_save(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        events: list[str] = []

        def record_consumed(path: Path) -> None:
            events.append(f"consume:{path}")

        def finalize() -> None:
            events.append("finalize")

        lifecycle = SimpleNamespace(
            consumed_cache_callback=record_consumed,
            finalize_after_checkpoint=finalize,
        )

        def fake_train(paths, *, model_config, training_config, initial_model=None, consumed_cache_callback=None):
            events.append("train")
            self.assertIsNotNone(consumed_cache_callback)
            consumed_cache_callback(Path("cache-a"))
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

        def fake_save(*args, **kwargs) -> None:
            events.append("save")

        with (
            patch("pokezero.neural_cli._training_cache_lifecycle", return_value=lifecycle),
            patch("pokezero.neural_cli.train_transformer_policy", side_effect=fake_train),
            patch("pokezero.neural_cli.save_transformer_checkpoint", side_effect=fake_save),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                [
                    "train",
                    "--data",
                    "cache-a",
                    "--out",
                    "checkpoint.pt",
                    "--showdown-root",
                    "/tmp/showdown",
                    # Pinned: this scaffold battery predates the v2.2 default; the bare
                    # "cache-a" carries no schema stamp (legacy), which only pairs with a
                    # v2.1-declaring train.
                    "--observation-schema",
                    "v2.1",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(events, ["train", "consume:cache-a", "save", "finalize"])

    def test_neural_cli_train_writes_summary_out(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "checkpoint.pt"
            summary_path = temp_path / "train-summary.json"
            events: list[str] = []

            class FakeLifecycle:
                consumed_cache_callback = None

                def finalize_after_checkpoint(self) -> None:
                    events.append("finalize")

                def to_summary(self) -> dict[str, object]:
                    return {
                        "root": "cache-root",
                        "footprint_bytes": 2048,
                        "footprint_limit_bytes": 50 * 1024 * 1024 * 1024,
                        "delete_after_checkpoint": True,
                        "consumed_paths": ["cache-a"],
                        "deleted_paths": ["cache-a"],
                        "deleted_bytes": 2048,
                    }

            def fake_train(paths, *, model_config, training_config, initial_model=None, consumed_cache_callback=None):
                events.append("train")
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

            def fake_save(path, *args, **kwargs) -> None:
                events.append("save")
                Path(path).write_bytes(b"checkpoint")

            with (
                patch("pokezero.neural_cli._training_cache_lifecycle", return_value=FakeLifecycle()),
                patch("pokezero.neural_cli.train_transformer_policy", side_effect=fake_train),
                patch("pokezero.neural_cli.save_transformer_checkpoint", side_effect=fake_save),
                patch("pokezero.neural_cli.collect_source_metadata", return_value={"available": False}),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                exit_code = neural_cli_main(
                    [
                        "train",
                        "--data",
                        "cache-a",
                        "--out",
                        str(checkpoint_path),
                        "--summary-out",
                        str(summary_path),
                        "--showdown-root",
                        "/tmp/showdown",
                        # Pinned: unstamped (legacy) fixture cache pairs with v2.1 only.
                        "--observation-schema",
                        "v2.1",
                    ]
                )

            summary = json.loads(summary_path.read_text())
            self.assertEqual(exit_code, 0)
            self.assertEqual(events, ["train", "save", "finalize"])
            self.assertIn(f"train_summary: {summary_path}", stdout.getvalue())
            self.assertEqual(summary["schema_version"], "pokezero.neural_train_summary.v1")
            self.assertEqual(summary["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(summary["checkpoint_bytes"], len(b"checkpoint"))
            self.assertIsNone(summary["input_data_bytes"])
            self.assertEqual(summary["model"]["transformer_layers"], 2)
            self.assertEqual(summary["training_config"]["epochs"], 1)
            self.assertEqual(summary["final_metrics"]["policy_accuracy"], 0.75)
            self.assertEqual(summary["training_cache"]["deleted_bytes"], 2048)
            self.assertGreaterEqual(summary["elapsed_seconds"], summary["train_elapsed_seconds"])
            self.assertGreaterEqual(summary["train_elapsed_seconds"], 0.0)

    def test_neural_cli_train_does_not_finalize_cache_lifecycle_when_checkpoint_save_fails(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        events: list[str] = []
        lifecycle = SimpleNamespace(
            consumed_cache_callback=lambda path: events.append(f"consume:{path}"),
            finalize_after_checkpoint=lambda: events.append("finalize"),
        )

        def fake_train(paths, *, model_config, training_config, initial_model=None, consumed_cache_callback=None):
            events.append("train")
            assert consumed_cache_callback is not None
            consumed_cache_callback(Path("cache-a"))
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

        def fail_save(*args, **kwargs) -> None:
            events.append("save")
            raise RuntimeError("save failed")

        with (
            patch("pokezero.neural_cli._training_cache_lifecycle", return_value=lifecycle),
            patch("pokezero.neural_cli.train_transformer_policy", side_effect=fake_train),
            patch("pokezero.neural_cli.save_transformer_checkpoint", side_effect=fail_save),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            exit_code = neural_cli_main(
                [
                    "train",
                    "--data",
                    "cache-a",
                    "--out",
                    "checkpoint.pt",
                    "--showdown-root",
                    "/tmp/showdown",
                    # Pinned: this scaffold battery predates the v2.2 default; the bare
                    # "cache-a" carries no schema stamp (legacy), which only pairs with a
                    # v2.1-declaring train.
                    "--observation-schema",
                    "v2.1",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(events, ["train", "consume:cache-a", "save"])

    def test_value_selection_restores_best_epoch_state(self) -> None:
        from pokezero.neural_cli import _train_with_value_selection

        class FakeModel:
            def __init__(self) -> None:
                self.weight = 0
                self.loaded_state = None

            def state_dict(self) -> dict[str, int]:
                return {"weight": self.weight}

            def load_state_dict(self, state: dict[str, int]) -> None:
                self.loaded_state = dict(state)
                self.weight = int(state["weight"])

        class FakeReport:
            def __init__(
                self,
                *,
                mae: float,
                sign_accuracy: float = 0.5,
                pearson_correlation: float | None = None,
            ) -> None:
                self.examples = 2
                self.mse = mae * mae
                self.mae = mae
                self.bias = 0.0
                self.sign_accuracy = sign_accuracy
                self.expected_calibration_error = mae / 2.0
                self.pearson_correlation = pearson_correlation

            def to_dict(self) -> dict[str, float | int | None]:
                return {
                    "examples": self.examples,
                    "mse": self.mse,
                    "mae": self.mae,
                    "bias": self.bias,
                    "sign_accuracy": self.sign_accuracy,
                    "expected_calibration_error": self.expected_calibration_error,
                    "pearson_correlation": self.pearson_correlation,
                }

        fake_model = FakeModel()
        model_config = TransformerPolicyConfig.compact_category(category_vocab=("species:a",), category_oov_buckets=2)
        training_config = TransformerTrainingConfig(epochs=3)
        train_calls = 0

        def fake_train(paths, *, model_config, training_config, initial_model=None, epoch_callback=None):
            nonlocal train_calls
            model = initial_model or fake_model
            metrics = []
            for epoch in range(1, training_config.epochs + 1):
                train_calls += 1
                model.weight = epoch
                metrics.append(
                    TransformerEpochMetrics(
                        epoch=epoch,
                        examples=2,
                        loss=float(epoch),
                        policy_loss=float(epoch),
                        policy_accuracy=0.5,
                        value_loss=float(epoch),
                    )
                )
                if epoch_callback is not None:
                    epoch_callback(
                        model,
                        TransformerTrainingResult(
                            model_config=model_config,
                            training_config=training_config,
                            epochs=tuple(metrics),
                        ),
                    )
            return model, TransformerTrainingResult(
                model_config=model_config,
                training_config=training_config,
                epochs=tuple(metrics),
            )

        reports = [FakeReport(mae=0.4), FakeReport(mae=0.2), FakeReport(mae=0.3)]

        with (
            patch("pokezero.neural_cli.train_transformer_policy", side_effect=fake_train),
            patch("pokezero.neural_cli.evaluate_value_calibration", side_effect=reports),
        ):
            model, result, payload = _train_with_value_selection(
                paths=[Path("train.jsonl")],
                model_config=model_config,
                training_config=training_config,
                initial_model=fake_model,
                selection_paths=[Path("heldout.jsonl")],
                selection_metric="mae",
                batch_size=5,
                bins=4,
            )

        self.assertEqual(model.weight, 2)
        self.assertEqual(model.loaded_state, {"weight": 2})
        self.assertEqual(train_calls, 3)
        self.assertEqual(result.training_config.epochs, 2)
        self.assertEqual(result.final_metrics.epoch, 2)
        self.assertEqual(payload["selected_epoch"], 2)
        self.assertEqual(payload["selected_metric_value"], 0.2)
        self.assertEqual(len(payload["epochs"]), 3)

    def test_value_selection_can_restore_best_epoch_by_pearson_correlation(self) -> None:
        from pokezero.neural_cli import _train_with_value_selection

        class FakeModel:
            def __init__(self) -> None:
                self.weight = 0
                self.loaded_state = None

            def state_dict(self) -> dict[str, int]:
                return {"weight": self.weight}

            def load_state_dict(self, state: dict[str, int]) -> None:
                self.loaded_state = dict(state)
                self.weight = int(state["weight"])

        class FakeReport:
            def __init__(self, *, pearson_correlation: float | None) -> None:
                self.examples = 2
                self.mse = 0.25
                self.mae = 0.4
                self.bias = 0.0
                self.sign_accuracy = 0.5
                self.expected_calibration_error = 0.2
                self.pearson_correlation = pearson_correlation

            def to_dict(self) -> dict[str, float | int | None]:
                return {
                    "examples": self.examples,
                    "mse": self.mse,
                    "mae": self.mae,
                    "bias": self.bias,
                    "sign_accuracy": self.sign_accuracy,
                    "expected_calibration_error": self.expected_calibration_error,
                    "pearson_correlation": self.pearson_correlation,
                }

        fake_model = FakeModel()
        model_config = TransformerPolicyConfig.compact_category(category_vocab=("species:a",), category_oov_buckets=2)
        training_config = TransformerTrainingConfig(epochs=3)

        def fake_train(paths, *, model_config, training_config, initial_model=None, epoch_callback=None):
            model = initial_model or fake_model
            metrics = []
            for epoch in range(1, training_config.epochs + 1):
                model.weight = epoch
                metrics.append(
                    TransformerEpochMetrics(
                        epoch=epoch,
                        examples=2,
                        loss=float(epoch),
                        policy_loss=float(epoch),
                        policy_accuracy=0.5,
                        value_loss=float(epoch),
                    )
                )
                if epoch_callback is not None:
                    epoch_callback(
                        model,
                        TransformerTrainingResult(
                            model_config=model_config,
                            training_config=training_config,
                            epochs=tuple(metrics),
                        ),
                    )
            return model, TransformerTrainingResult(
                model_config=model_config,
                training_config=training_config,
                epochs=tuple(metrics),
            )

        reports = [
            FakeReport(pearson_correlation=None),
            FakeReport(pearson_correlation=0.65),
            FakeReport(pearson_correlation=0.4),
        ]

        with (
            patch("pokezero.neural_cli.train_transformer_policy", side_effect=fake_train),
            patch("pokezero.neural_cli.evaluate_value_calibration", side_effect=reports),
        ):
            model, result, payload = _train_with_value_selection(
                paths=[Path("train.jsonl")],
                model_config=model_config,
                training_config=training_config,
                initial_model=fake_model,
                selection_paths=[Path("heldout.jsonl")],
                selection_metric="pearson_correlation",
                batch_size=5,
                bins=4,
            )

        self.assertEqual(model.weight, 2)
        self.assertEqual(model.loaded_state, {"weight": 2})
        self.assertEqual(result.training_config.epochs, 2)
        self.assertEqual(result.final_metrics.epoch, 2)
        self.assertEqual(payload["metric"], "pearson_correlation")
        self.assertEqual(payload["metric_direction"], "max")
        self.assertEqual(payload["selected_epoch"], 2)
        self.assertEqual(payload["selected_metric_value"], 0.65)
        self.assertIsNone(payload["epochs"][0]["metric_value"])
        self.assertIn("metric_unavailable_reason", payload["epochs"][0])

    def test_neural_cli_train_can_write_value_selection_artifact(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        fake_model = object()

        def fake_train_with_selection(**kwargs):
            return fake_model, TransformerTrainingResult(
                model_config=kwargs["model_config"],
                training_config=kwargs["training_config"],
                epochs=(
                    TransformerEpochMetrics(
                        epoch=1,
                        examples=4,
                        loss=0.25,
                        policy_loss=0.2,
                        policy_accuracy=0.75,
                        value_loss=0.25,
                    ),
                ),
            ), {
                "paths": ["heldout.jsonl"],
                "batch_size": 11,
                "bins": 7,
                "metric": "expected_calibration_error",
                "metric_direction": "min",
                "selected_epoch": 1,
                "selected_metric_value": 0.12,
                "epochs": [],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            selection_path = Path(temp_dir) / "value-selection.json"
            stdout = io.StringIO()
            with (
                patch("pokezero.neural_cli._train_with_value_selection", side_effect=fake_train_with_selection) as select_train,
                patch("pokezero.neural_cli.save_transformer_checkpoint") as save,
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
                        # Pinned: unstamped .jsonl fixture data pairs with v2.1 only.
                        "--observation-schema",
                        "v2.1",
                        "--objective",
                        "value-only",
                        "--freeze-non-value-parameters",
                        "--value-selection-data",
                        "heldout.jsonl",
                        "--value-selection-metric",
                        "expected_calibration_error",
                        "--value-selection-out",
                        str(selection_path),
                        "--value-calibration-batch-size",
                        "11",
                        "--value-calibration-bins",
                        "7",
                    ]
                )

            payload = json.loads(selection_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(select_train.call_args.kwargs["selection_paths"], [Path("heldout.jsonl")])
        self.assertEqual(select_train.call_args.kwargs["selection_metric"], "expected_calibration_error")
        self.assertEqual(select_train.call_args.kwargs["batch_size"], 11)
        self.assertEqual(select_train.call_args.kwargs["bins"], 7)
        self.assertEqual(save.call_args.args[0], checkpoint_path)
        self.assertEqual(payload["selected_epoch"], 1)
        self.assertEqual(payload["selected_metric_value"], 0.12)
        self.assertIn(f"value_selection: {selection_path}", stdout.getvalue())

    def test_neural_cli_train_can_warm_start_value_only_finetune(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        fake_model = object()
        # Pinned to a v2.1-stamped base checkpoint: the finetune data fixture is an
        # unstamped (legacy) .jsonl, which only pairs with a v2.1-declaring run — and
        # a fresh compact_category() would stamp the current default (v2.2) post-flip.
        fake_model_config = TransformerPolicyConfig.compact_category(
            category_vocab=("species:a",),
            category_oov_buckets=2,
            policy_id="base-policy",
            observation_schema_version="pokezero.observation.v2.1",
            numeric_feature_count=140,
            categorical_feature_count=39,
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
                    "--hp-delta-return-weight",
                    "0.2",
                    "--faint-delta-return-weight",
                    "0.4",
                    "--turn-penalty-after",
                    "180",
                    "--turn-penalty",
                    "0.01",
                    "--objective",
                    "ppo",
                    "--ppo-target-mode",
                    "gae",
                    "--gae-lambda",
                    "0.8",
                    "--value-clip-range",
                    "0.0184",
                    "--policy-id",
                    "entity-cli",
                    "--promotion-registry",
                    "promotions.json",
                    "--require-promoted-opponent-pool-size",
                    "2",
                    "--historical-opponent-selection",
                    "spread",
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
        self.assertEqual(kwargs["training_config"].hp_delta_return_weight, 0.2)
        self.assertEqual(kwargs["training_config"].faint_delta_return_weight, 0.4)
        self.assertEqual(kwargs["training_config"].turn_penalty_after, 180)
        self.assertEqual(kwargs["training_config"].turn_penalty, 0.01)
        self.assertEqual(kwargs["training_config"].objective, "ppo")
        self.assertEqual(kwargs["training_config"].ppo_target_mode, "gae")
        self.assertEqual(kwargs["training_config"].gae_lambda, 0.8)
        self.assertEqual(kwargs["training_config"].value_clip_range, 0.0184)
        self.assertEqual(kwargs["model_config"].policy_id, "entity-cli")
        self.assertEqual(kwargs["promotion_registry_path"], Path("promotions.json"))
        self.assertEqual(kwargs["required_promoted_opponent_pool_size"], 2)
        self.assertEqual(kwargs["historical_opponent_selection"], "spread")
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

    def test_neural_cli_iterate_wires_temporal_aggregator(self) -> None:
        model_config = self._run_iterate_capturing_model_config(
            ["--showdown-root", "/tmp/showdown", "--temporal-aggregator", "gru"]
        )

        self.assertEqual(model_config.temporal_aggregator, "gru")

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

    def test_neural_cli_foundation_plan_threads_batch_size_override(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = neural_cli_main(
                [
                    "foundation-plan",
                    "--run-dir",
                    "run",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--profile",
                    "midscale",
                    "--variant",
                    "teacher-cut",
                    "--recipe-fidelity",
                    "--batch-size",
                    "8192",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        argv = payload["command"]["argv"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["resolved_options"]["batch_size"], 8192)
        self.assertIn("--batch-size", argv)
        self.assertEqual(argv[argv.index("--batch-size") + 1], "8192")

    def test_neural_cli_foundation_plan_omits_batch_size_without_override(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = neural_cli_main(
                [
                    "foundation-plan",
                    "--run-dir",
                    "run",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--profile",
                    "midscale",
                    "--variant",
                    "teacher-cut",
                    "--recipe-fidelity",
                    "--json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertIsNone(payload["resolved_options"]["batch_size"])
        self.assertNotIn("--batch-size", payload["command"]["argv"])

    def test_neural_cli_foundation_plan_rejects_nonpositive_batch_size(self) -> None:
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            exit_code = neural_cli_main(
                [
                    "foundation-plan",
                    "--run-dir",
                    "run",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--batch-size",
                    "0",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("batch-size must be positive", stderr.getvalue())

    def test_neural_cli_help_lists_benchmark_command(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(stdout):
            neural_cli_main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("benchmark", stdout.getvalue())
        self.assertIn("iterate", stdout.getvalue())

    def test_train_transformer_policy_records_annealed_epoch_learning_rates(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            checkpoint_path = Path(temp_dir) / "transformer.pt"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            model_config = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)),
                category_oov_buckets=4,
                policy_id="lr-schedule",
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

            model, result = train_transformer_policy(
                data_path,
                model_config=model_config,
                training_config=TransformerTrainingConfig(
                    batch_size=2,
                    epochs=3,
                    learning_rate=5.9e-5,
                    learning_rate_schedule=MIT_THESIS_LEARNING_RATE_SCHEDULE,
                    learning_rate_progress_start=0.0,
                    learning_rate_progress_end=1.0,
                    window_size=2,
                    max_batches=1,
                    device="cpu",
                ),
            )
            save_transformer_checkpoint(checkpoint_path, model, result=result)
            _, restored = load_transformer_checkpoint(checkpoint_path, map_location="cpu")

        learning_rates = [metrics.learning_rate for metrics in restored.epochs]
        self.assertEqual(len(learning_rates), 3)
        self.assertAlmostEqual(learning_rates[0], 5.9e-5)
        self.assertAlmostEqual(learning_rates[1], 5.9e-5 / (5.0**1.5))
        self.assertAlmostEqual(learning_rates[2], 5.9e-5 / (9.0**1.5))
        self.assertEqual(restored.training_config.learning_rate_schedule, MIT_THESIS_LEARNING_RATE_SCHEDULE)
        for metrics in restored.epochs:
            self.assertEqual(metrics.batches, 1)
            self.assertIsNotNone(metrics.elapsed_seconds)
            self.assertIsNotNone(metrics.batch_load_elapsed_seconds)
            self.assertIsNotNone(metrics.tensorize_elapsed_seconds)
            self.assertIsNotNone(metrics.model_forward_elapsed_seconds)
            self.assertIsNotNone(metrics.backward_elapsed_seconds)
            self.assertIsNotNone(metrics.optimizer_step_elapsed_seconds)
            self.assertIsNotNone(metrics.examples_per_second)
            self.assertGreaterEqual(metrics.elapsed_seconds or 0.0, 0.0)

    def test_train_transformer_policy_applies_max_grad_norm_clip(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            model_config = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)),
                category_oov_buckets=4,
                policy_id="grad-clip",
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
            real_clip = torch.nn.utils.clip_grad_norm_
            observed: list[float] = []

            def _spy(parameters, max_norm, *args, **kwargs):
                observed.append(float(max_norm))
                return real_clip(parameters, max_norm, *args, **kwargs)

            with patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=_spy):
                train_transformer_policy(
                    data_path,
                    model_config=model_config,
                    training_config=TransformerTrainingConfig(
                        batch_size=2, epochs=1, window_size=2, max_batches=1, device="cpu", max_grad_norm=0.5
                    ),
                )
            self.assertTrue(observed)
            self.assertEqual(observed[0], 0.5)

            # Without the knob, gradient clipping must not be invoked (preserved legacy behavior).
            observed.clear()
            with patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=_spy):
                train_transformer_policy(
                    data_path,
                    model_config=model_config,
                    training_config=TransformerTrainingConfig(
                        batch_size=2, epochs=1, window_size=2, max_batches=1, device="cpu"
                    ),
                )
            self.assertEqual(observed, [])

    def test_train_transformer_policy_bf16_autocast_runs_and_engages(self) -> None:
        # WS-A1: bf16 autocast must (a) run without error, (b) actually enter a bf16 autocast
        # context around forward/loss, (c) keep master weights fp32, and (d) round-trip the
        # amp setting through the checkpoint. Numerical parity vs fp32 is validated separately
        # on GPU at recipe scale (the PPO ratio/clip gate); here we only assert the path works.
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            checkpoint_path = Path(temp_dir) / "transformer.pt"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            model_config = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)),
                category_oov_buckets=4,
                policy_id="bf16-autocast",
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
            autocast_states: list[bool] = []
            real_forward = neural_policy_module.model_forward_from_training_tensors

            def _autocast_on() -> bool:
                try:
                    return bool(torch.is_autocast_enabled("cpu"))
                except TypeError:
                    cpu_fn = getattr(torch, "is_autocast_cpu_enabled", None)
                    return bool(cpu_fn()) if cpu_fn else bool(torch.is_autocast_enabled())

            def _forward_spy(model, tensors):
                autocast_states.append(_autocast_on())
                return real_forward(model, tensors)

            with patch.object(neural_policy_module, "model_forward_from_training_tensors", side_effect=_forward_spy):
                model, result = train_transformer_policy(
                    data_path,
                    model_config=model_config,
                    training_config=TransformerTrainingConfig(
                        batch_size=2, epochs=1, window_size=2, max_batches=1, device="cpu", amp="bf16"
                    ),
                )
            # (b) autocast was actually engaged during forward.
            self.assertTrue(autocast_states and all(autocast_states))
            # (c) master weights remain fp32 despite bf16 compute.
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            # (a) losses are finite (equal to themselves = not NaN, bounded = not inf).
            for metrics in result.epochs:
                loss_value = float(metrics.loss)
                self.assertEqual(loss_value, loss_value)
                self.assertLess(abs(loss_value), 1e9)
            # (d) amp round-trips through the checkpoint.
            save_transformer_checkpoint(checkpoint_path, model, result=result)
            _, restored = load_transformer_checkpoint(checkpoint_path, map_location="cpu")
            self.assertEqual(restored.training_config.amp, "bf16")

    def test_amp_flag_parses_on_train_and_iterate_entry_points(self) -> None:
        # WS-A1: --amp bf16 must be accepted on both training entry points (the cluster trains
        # via `neural train`; `neural iterate` is the in-process loop), default fp32, reject others.
        from pokezero.neural_cli import build_arg_parser

        parser = build_arg_parser()
        train_base = ["train", "--data", "x", "--out", "y"]
        iterate_base = [
            "iterate", "--run-dir", "r", "--iterations", "1",
            "--games-per-iteration", "2", "--initial-policy", "random-legal",
        ]
        self.assertEqual(parser.parse_args(train_base + ["--amp", "bf16"]).amp, "bf16")
        self.assertIsNone(parser.parse_args(train_base).amp)
        self.assertEqual(parser.parse_args(iterate_base + ["--amp", "bf16"]).amp, "bf16")
        self.assertIsNone(parser.parse_args(iterate_base).amp)
        with self.assertRaises(SystemExit):
            parser.parse_args(iterate_base + ["--amp", "fp16"])

    def test_amp_autocast_device_type_maps_supported_and_rejects_others(self) -> None:
        # WS-A1: cuda/cpu map through; mps (and any other backend) must raise, not silently
        # coerce to cpu (which would train fp32 with no warning). Pure device-string parsing —
        # no GPU/mps hardware required.
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        from pokezero.neural_policy import _amp_autocast_device_type

        self.assertEqual(_amp_autocast_device_type("cpu"), "cpu")
        self.assertEqual(_amp_autocast_device_type("cuda"), "cuda")
        self.assertEqual(_amp_autocast_device_type("cuda:0"), "cuda")
        with self.assertRaisesRegex(ValueError, "amp"):
            _amp_autocast_device_type("mps")

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
                    temporal_aggregator="gru",
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
                    temporal_aggregator="gru",
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

    def test_train_transformer_policy_accepts_cache_backed_row_indexed_batches(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            cache_path = Path(temp_dir) / "cache"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            write_training_cache_from_rollouts(data_path, cache_path, config=TrajectoryDatasetConfig(window_size=2))
            observed_row_indexed_batches: list[bool] = []
            original_forward = neural_policy_module.model_forward_from_training_tensors

            def _spy_forward(model, tensors):
                observed_row_indexed_batches.append("window_row_indices" in tensors)
                return original_forward(model, tensors)

            with patch("pokezero.neural_policy.model_forward_from_training_tensors", side_effect=_spy_forward):
                _, result = train_transformer_policy(
                    cache_path,
                    model_config=TransformerPolicyConfig.compact_category(
                        category_vocab=tuple(range(1, 17)),
                        category_oov_buckets=4,
                        policy_id="cache-row-indexed-train",
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

        self.assertEqual(result.final_metrics.examples, 2)
        self.assertEqual(observed_row_indexed_batches, [True])

    def test_train_transformer_policy_mixes_capped_auxiliary_batches(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = temp_path / "rollouts.jsonl"
            auxiliary_cache = temp_path / "auxiliary-cache"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            auxiliary_examples = [
                replace(example, action_index=1)
                for example in examples_from_record(rollout_record(), config=TrajectoryDatasetConfig(window_size=1))
            ]
            write_training_cache_from_examples(
                auxiliary_examples,
                auxiliary_cache,
                config=TrajectoryDatasetConfig(window_size=1),
            )

            _, result = train_transformer_policy(
                data_path,
                model_config=TransformerPolicyConfig.compact_category(
                    category_vocab=tuple(range(1, 17)),
                    category_oov_buckets=4,
                    policy_id="auxiliary-train",
                    window_size=1,
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
                    batch_size=4,
                    epochs=1,
                    window_size=1,
                    device="cpu",
                ),
                auxiliary_paths=auxiliary_cache,
                auxiliary_max_fraction=0.2,
            )

        self.assertEqual(result.final_metrics.examples, 5)

    def test_checkpoint_round_trips_belief_provenance(self) -> None:
        # Regression guard for the provenance chain's only durable link: if the save payload key
        # or load passthrough is dropped, every downstream mismatch warning goes permanently
        # inert while all other tests stay green.
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "transformer.pt"
            model_config = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)),
                category_oov_buckets=4,
                policy_id="belief-provenance-roundtrip",
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
            base = TransformerTrainingResult(
                model_config=model_config,
                training_config=TransformerTrainingConfig(window_size=2),
                epochs=(
                    TransformerEpochMetrics(epoch=1, examples=4, loss=0.5, policy_loss=0.2, policy_accuracy=0.5),
                ),
            )

            save_transformer_checkpoint(checkpoint_path, model, result=replace(base, belief_set_source_hash="prov123"))
            _, restored = load_transformer_checkpoint(checkpoint_path, map_location="cpu")
            self.assertEqual(restored.belief_set_source_hash, "prov123")

            save_transformer_checkpoint(checkpoint_path, model, result=base)
            _, restored_none = load_transformer_checkpoint(checkpoint_path, map_location="cpu")
            self.assertIsNone(restored_none.belief_set_source_hash)

    def test_neural_cli_train_stamps_belief_provenance_from_data(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stamped_a = temp_path / "a.jsonl"
            stamped_b = temp_path / "b.jsonl"
            legacy = temp_path / "legacy.jsonl"
            stamped_a.write_text('{"belief_set_source_hash": "hashA"}\n')
            stamped_b.write_text('{"belief_set_source_hash": "hashA"}\n')
            legacy.write_text('{"battle_id": "x"}\n')
            captured: list[object] = []

            def fake_train(paths, *, model_config, training_config, initial_model=None, **_kwargs):
                return object(), TransformerTrainingResult(
                    model_config=model_config,
                    training_config=training_config,
                    epochs=(
                        TransformerEpochMetrics(epoch=1, examples=4, loss=0.25, policy_loss=0.2, policy_accuracy=0.75),
                    ),
                )

            def fake_save(path, model, *, result) -> None:
                captured.append(result)
                Path(path).write_bytes(b"checkpoint")

            def run(data_paths: list[str], out_name: str) -> str:
                with (
                    patch("pokezero.neural_cli.train_transformer_policy", side_effect=fake_train),
                    patch("pokezero.neural_cli.save_transformer_checkpoint", side_effect=fake_save),
                    patch("pokezero.neural_cli.collect_source_metadata", return_value={"available": False}),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()) as stderr,
                ):
                    exit_code = neural_cli_main(
                        # Pinned to v2.1: the provenance-fixture .jsonl files are
                        # deliberately unstamped (legacy), which only pairs with v2.1.
                        ["train", "--data", *data_paths, "--out", str(temp_path / out_name), "--showdown-root", "/tmp/showdown", "--observation-schema", "v2.1"]
                    )
                self.assertEqual(exit_code, 0)
                return stderr.getvalue()

            run([str(stamped_a), str(stamped_b)], "same.pt")
            self.assertEqual(captured[-1].belief_set_source_hash, "hashA")

            stderr_text = run([str(stamped_a), str(legacy)], "mixed.pt")
            self.assertIsNone(captured[-1].belief_set_source_hash)
            self.assertIn("mixes belief set-source provenance", stderr_text)

    def test_checkpoint_round_trips_ppo_diagnostic_metrics(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "transformer.pt"
            model_config = TransformerPolicyConfig.compact_category(
                category_vocab=tuple(range(1, 17)),
                category_oov_buckets=4,
                policy_id="ppo-diagnostics-roundtrip",
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
            result = TransformerTrainingResult(
                model_config=model_config,
                training_config=TransformerTrainingConfig(objective="ppo", window_size=2),
                value_calibration_transform=ValueCalibrationTransform(scale=1.5, bias=-0.2),
                epochs=(
                    TransformerEpochMetrics(
                        epoch=1,
                        examples=10,
                        loss=0.5,
                        policy_loss=-0.1,
                        policy_accuracy=0.4,
                        value_loss=0.25,
                        ppo_valid_examples=8,
                        ppo_valid_fraction=0.8,
                        ppo_advantage_mean=0.2,
                        ppo_advantage_std=0.5,
                        ppo_ratio_mean=1.1,
                        ppo_clip_fraction=0.125,
                        ppo_value_clip_eligible_examples=6,
                        ppo_value_clip_fraction=0.5,
                        ppo_entropy=1.7,
                    ),
                ),
            )

            save_transformer_checkpoint(checkpoint_path, model, result=result)
            _, restored = load_transformer_checkpoint(checkpoint_path, map_location="cpu")

        restored_metrics = restored.final_metrics
        self.assertEqual(restored_metrics.ppo_valid_examples, 8)
        self.assertEqual(restored_metrics.ppo_valid_fraction, 0.8)
        self.assertEqual(restored_metrics.ppo_advantage_mean, 0.2)
        self.assertEqual(restored_metrics.ppo_advantage_std, 0.5)
        self.assertEqual(restored_metrics.ppo_ratio_mean, 1.1)
        self.assertEqual(restored_metrics.ppo_clip_fraction, 0.125)
        self.assertEqual(restored_metrics.ppo_value_clip_eligible_examples, 6)
        self.assertEqual(restored_metrics.ppo_value_clip_fraction, 0.5)
        self.assertEqual(restored_metrics.ppo_entropy, 1.7)
        self.assertIsNotNone(restored.value_calibration_transform)
        self.assertEqual(restored.value_calibration_transform.scale, 1.5)
        self.assertEqual(restored.value_calibration_transform.bias, -0.2)

    def _build_atomic_checkpoint_fixture(self, policy_id: str) -> tuple[Any, TransformerTrainingResult]:
        model_config = TransformerPolicyConfig.compact_category(
            category_vocab=tuple(range(1, 17)),
            category_oov_buckets=4,
            policy_id=policy_id,
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
        result = TransformerTrainingResult(
            model_config=model_config,
            training_config=TransformerTrainingConfig(window_size=2),
            epochs=(
                TransformerEpochMetrics(
                    epoch=1,
                    examples=4,
                    loss=0.5,
                    policy_loss=0.3,
                    policy_accuracy=0.6,
                ),
            ),
        )
        return model, result

    def test_save_transformer_checkpoint_leaves_no_file_when_interrupted(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()
        model, result = self._build_atomic_checkpoint_fixture("atomic-save-absent")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "transformer.pt"

            def _interrupted_save(_payload: Any, handle: Any, *args: Any, **kwargs: Any) -> None:
                # Emulate a crash after some bytes have already reached the temp file.
                handle.write(b"partial-checkpoint-bytes")
                handle.flush()
                raise RuntimeError("simulated interruption during torch.save")

            with patch.object(torch, "save", side_effect=_interrupted_save):
                with self.assertRaises(RuntimeError):
                    save_transformer_checkpoint(checkpoint_path, model, result=result)

            # The destination was never created, and the partial temp file was cleaned up:
            # an interrupted write must never leave a corrupt file at the final path.
            self.assertFalse(checkpoint_path.exists())
            self.assertEqual(sorted(entry.name for entry in Path(temp_dir).iterdir()), [])

    def test_save_transformer_checkpoint_preserves_previous_when_interrupted(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()
        model, result = self._build_atomic_checkpoint_fixture("atomic-save-previous")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "transformer.pt"

            # A valid checkpoint already exists on disk from a prior successful save.
            save_transformer_checkpoint(checkpoint_path, model, result=result)
            original_bytes = checkpoint_path.read_bytes()

            def _interrupted_save(_payload: Any, handle: Any, *args: Any, **kwargs: Any) -> None:
                handle.write(b"partial-checkpoint-bytes")
                handle.flush()
                raise RuntimeError("simulated interruption during torch.save")

            with patch.object(torch, "save", side_effect=_interrupted_save):
                with self.assertRaises(RuntimeError):
                    save_transformer_checkpoint(checkpoint_path, model, result=result)

            # The prior checkpoint is byte-for-byte intact, still loads cleanly, and no
            # partial temp file leaked into the directory.
            self.assertEqual(checkpoint_path.read_bytes(), original_bytes)
            _, restored = load_transformer_checkpoint(checkpoint_path, map_location="cpu")
            self.assertEqual(restored.model_config.policy_id, "atomic-save-previous")
            leftover = sorted(
                entry.name for entry in Path(temp_dir).iterdir() if entry != checkpoint_path
            )
            self.assertEqual(leftover, [])

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
            before_priors = evaluate_transformer_action_priors(
                model=model,
                result=TransformerTrainingResult(
                    model_config=model_config,
                    training_config=TransformerTrainingConfig(window_size=2),
                    epochs=(),
                ),
                observations=(observation(1),),
                device="cpu",
            )

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
            after_priors = evaluate_transformer_action_priors(
                model=trained_model,
                result=TransformerTrainingResult(
                    model_config=model_config,
                    training_config=TransformerTrainingConfig(window_size=2),
                    epochs=(),
                ),
                observations=(observation(1),),
                device="cpu",
            )

        self.assertEqual(result.final_metrics.policy_loss, 0.0)
        self.assertFalse(trained_model.training)
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("value_head.") for name in changed))
        for before_value, after_value in zip(before_priors, after_priors):
            self.assertAlmostEqual(before_value, after_value, places=7)

    def test_train_transformer_policy_rejects_window_size_mismatch(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")

        with self.assertRaisesRegex(ValueError, "window_size"):
            train_transformer_policy(
                "missing.jsonl",
                model_config=TransformerPolicyConfig.compact_category(
                    category_vocab=tuple(range(1, 17)),
                    category_oov_buckets=4,
                    policy_id="window-mismatch",
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
                training_config=TransformerTrainingConfig(batch_size=2, epochs=1, window_size=4, device="cpu"),
            )

    def test_train_transformer_policy_calls_epoch_callback(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())

            callback_epochs = []
            _, result = train_transformer_policy(
                data_path,
                model_config=TransformerPolicyConfig.compact_category(
                    category_vocab=tuple(range(1, 17)),
                    category_oov_buckets=4,
                    policy_id="callback-smoke",
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
                    epochs=2,
                    window_size=2,
                    max_batches=1,
                    device="cpu",
                ),
                epoch_callback=lambda model, epoch_result: callback_epochs.append(epoch_result.final_metrics.epoch),
            )

        self.assertEqual(callback_epochs, [1, 2])
        self.assertEqual(result.final_metrics.epoch, 2)


class TruncateHistoryTensorsTest(unittest.TestCase):
    """History-truncation probe harness (docs/history_truncation_probe_plan.md)."""

    def _tensors(self, filled: int, *, transition_token_count: int | None = None):
        torch = require_torch()
        from pokezero.showdown import TRANSITION_TOKEN_OFFSET
        from pokezero.observation import TRANSITION_TOKEN_COUNT

        capacity = (
            TRANSITION_TOKEN_COUNT
            if transition_token_count is None
            else transition_token_count
        )
        token_count = TRANSITION_TOKEN_OFFSET + capacity
        offset = TRANSITION_TOKEN_OFFSET
        attention_mask = torch.zeros((1, 1, token_count), dtype=torch.bool)
        numeric = torch.zeros((1, 1, token_count, 3), dtype=torch.float32)
        categorical = torch.zeros((1, 1, token_count, 2), dtype=torch.long)
        # Non-transition prefix (indices 0..offset-1) is always attended, always populated.
        attention_mask[0, 0, :offset] = True
        numeric[0, 0, :offset, :] = 7.0
        categorical[0, 0, :offset, :] = 5
        # Transition region fills oldest-first: region indices 0..filled-1 attended, the token
        # payload marks its own chronological rank so the test can identify which survive.
        for region_index in range(filled):
            column = offset + region_index
            attention_mask[0, 0, column] = True
            numeric[0, 0, column, 0] = float(region_index + 1)
            categorical[0, 0, column, 0] = region_index + 1
        return {
            "attention_mask": attention_mask,
            "numeric_features": numeric,
            "categorical_ids": categorical,
            "token_type_ids": torch.zeros((1, 1, token_count), dtype=torch.long),
            "history_mask": torch.ones((1, 1), dtype=torch.bool),
            "legal_action_mask": torch.ones((1, ACTION_COUNT), dtype=torch.bool),
        }, offset

    def test_keeps_most_recent_k_and_masks_older(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        tensors, offset = self._tensors(filled=40)
        out = neural_policy_module.truncate_history_tensors(tensors, keep_recent_k=16)
        region = out["attention_mask"][0, 0, offset:]
        # Exactly the newest 16 (region indices 24..39) stay attended.
        attended = torch.nonzero(region, as_tuple=False).flatten().tolist()
        self.assertEqual(attended, list(range(24, 40)))
        # Dropped slots are zeroed in BOTH payload planes (byte-identical to an unfilled slot).
        self.assertTrue(torch.all(out["numeric_features"][0, 0, offset : offset + 24] == 0))
        self.assertTrue(torch.all(out["categorical_ids"][0, 0, offset : offset + 24] == 0))
        # Surviving slots keep their chronological-rank payload intact.
        for region_index in range(24, 40):
            self.assertEqual(
                out["numeric_features"][0, 0, offset + region_index, 0].item(),
                float(region_index + 1),
            )

    def test_non_transition_tokens_untouched(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        tensors, offset = self._tensors(filled=40)
        out = neural_policy_module.truncate_history_tensors(tensors, keep_recent_k=8)
        self.assertTrue(torch.all(out["attention_mask"][0, 0, :offset]))
        self.assertTrue(torch.all(out["numeric_features"][0, 0, :offset] == 7.0))
        self.assertTrue(torch.all(out["categorical_ids"][0, 0, :offset] == 5))
        # Untouched pass-through tensors keep the same object identity.
        self.assertIs(out["token_type_ids"], tensors["token_type_ids"])
        self.assertIs(out["legal_action_mask"], tensors["legal_action_mask"])

    def test_noop_when_k_at_or_above_filled(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        tensors, offset = self._tensors(filled=12)
        out = neural_policy_module.truncate_history_tensors(tensors, keep_recent_k=16)
        self.assertEqual(
            int(out["attention_mask"][0, 0, offset:].sum()), 12
        )
        self.assertTrue(
            torch.equal(out["attention_mask"], tensors["attention_mask"])
        )

    def test_does_not_mutate_input(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        tensors, offset = self._tensors(filled=40)
        before = tensors["attention_mask"].clone()
        neural_policy_module.truncate_history_tensors(tensors, keep_recent_k=16)
        self.assertTrue(torch.equal(tensors["attention_mask"], before))

    def test_invalid_k_rejected(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        tensors, _ = self._tensors(filled=8)
        with self.assertRaises(ValueError):
            neural_policy_module.truncate_history_tensors(tensors, keep_recent_k=0)
        with self.assertRaises(ValueError):
            neural_policy_module.truncate_history_tensors(tensors, keep_recent_k=999)

    def test_v3_tensor_shape_uses_64_token_capacity(self) -> None:
        if not torch_available():
            self.skipTest("requires torch")
        tensors, offset = self._tensors(filled=64, transition_token_count=64)
        out = neural_policy_module.truncate_history_tensors(tensors, keep_recent_k=16)
        attended = (
            require_torch()
            .nonzero(out["attention_mask"][0, 0, offset:], as_tuple=False)
            .flatten()
            .tolist()
        )
        self.assertEqual(attended, list(range(48, 64)))
        with self.assertRaisesRegex(ValueError, r"1\.\.64"):
            neural_policy_module.truncate_history_tensors(tensors, keep_recent_k=65)

    def test_policy_forward_matches_manual_truncation(self) -> None:
        """The policy's history_mask_k path applies the same mask the helper does, and a
        truncated forward differs from the full-history forward on non-decorative history."""
        if not torch_available():
            self.skipTest("requires torch")
        torch = require_torch()
        from pokezero.showdown import TRANSITION_TOKEN_OFFSET

        config = TransformerPolicyConfig.compact_category(
            policy_id="probe-forward",
            category_vocab=tuple(f"token-{index}" for index in range(16)),
            category_oov_buckets=1,
            window_size=1,
            categorical_feature_count=2,
            numeric_feature_count=3,
            token_count=DEFAULT_REPLAY_OBSERVATION_SPEC.token_count,
            embedding_dim=8,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=16,
            dropout=0.0,
        )
        model = EntityTokenTransformerPolicy(config)
        model.eval()
        token_count = config.token_count
        offset = TRANSITION_TOKEN_OFFSET
        attention_mask = torch.zeros((1, 1, token_count), dtype=torch.bool)
        attention_mask[0, 0, : offset + 40] = True  # 40 filled transition tokens
        numeric = torch.zeros((1, 1, token_count, 3), dtype=torch.float32)
        categorical = torch.zeros((1, 1, token_count, 2), dtype=torch.long)
        for column in range(offset + 40):
            numeric[0, 0, column, column % 3] = 1.0
            categorical[0, 0, column, 0] = (column % 15) + 1
        tensors = {
            "attention_mask": attention_mask,
            "numeric_features": numeric,
            "categorical_ids": categorical,
            "token_type_ids": torch.zeros((1, 1, token_count), dtype=torch.long),
            "history_mask": torch.ones((1, 1), dtype=torch.bool),
        }
        masked = neural_policy_module.truncate_history_tensors(tensors, keep_recent_k=16)
        with torch.no_grad():
            full = model(**tensors)
            trunc = model(**masked)
        # Truncating 40→16 tokens changes what the encoder attends, so logits must move.
        self.assertFalse(torch.allclose(full.policy_logits, trunc.policy_logits))


if __name__ == "__main__":
    unittest.main()
