import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import pokezero.linear_policy as linear_policy_module
from pokezero.actions import ACTION_COUNT, ACTION_SCHEMA_VERSION
from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.env import TerminalState
from pokezero.linear_cli import main as linear_cli_main
from pokezero.linear_policy import (
    LINEAR_FEATURE_SCHEMA_VERSION,
    LINEAR_POLICY_SCHEMA_VERSION,
    LinearPolicyModel,
    LinearSoftmaxPolicy,
    LinearTrainingConfig,
    evaluate_linear_policy,
    features_from_observation_window,
    features_from_window,
    linear_feature_fingerprint,
    load_linear_model,
    save_linear_model,
    train_linear_policy,
    _linear_feature_fingerprint_payload,
)
from pokezero.observation import OBSERVATION_SCHEMA_VERSION, ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


LEGAL_TWO_ACTION_MASK = (True, True, False, False, False, False, False, False, False)


def observation(value: int, *, player_id: str = "p1") -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((value,) for _ in range(spec.token_count)),
        numeric_features=tuple((float(value),) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=LEGAL_TWO_ACTION_MASK,
        perspective=ObservationPerspective.from_showdown_slot(player_id, "p1"),
    )


def separable_record() -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id="linear-train", format_id="gen3randombattle", seed=1)
    for turn_index in range(24):
        action_index = turn_index % 2
        value = 10 if action_index == 0 else 30
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=turn_index,
                observation=observation(value),
                legal_action_mask=LEGAL_TWO_ACTION_MASK,
                action_index=action_index,
            )
        )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=24))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "oracle"},
        decision_round_count=24,
        elapsed_seconds=0.1,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def opponent_action_record() -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id="opponent-action-train", format_id="gen3randombattle", seed=5)
    for turn_index in range(24):
        opponent_action_index = turn_index % 2
        value = 10 if opponent_action_index == 0 else 30
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=turn_index,
                observation=observation(value),
                legal_action_mask=LEGAL_TWO_ACTION_MASK,
                action_index=0,
                opponent_action_index=opponent_action_index,
            )
        )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=24))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "oracle"},
        decision_round_count=24,
        elapsed_seconds=0.1,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def losing_action_record() -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id="linear-losing", format_id="gen3randombattle", seed=2)
    trajectory.append(
        TrajectoryStep(
            player_id="p1",
            turn_index=0,
            observation=observation(10),
            legal_action_mask=LEGAL_TWO_ACTION_MASK,
            action_index=0,
        )
    )
    trajectory.record_terminal(TerminalState(winner="p2", turn_count=1))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "loser"},
        decision_round_count=1,
        elapsed_seconds=0.1,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def winning_action_record() -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id="linear-winning", format_id="gen3randombattle", seed=3)
    trajectory.append(
        TrajectoryStep(
            player_id="p1",
            turn_index=0,
            observation=observation(10),
            legal_action_mask=LEGAL_TWO_ACTION_MASK,
            action_index=1,
        )
    )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=1))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "winner"},
        decision_round_count=1,
        elapsed_seconds=0.1,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def capped_action_record() -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id="linear-capped", format_id="gen3randombattle", seed=4)
    trajectory.append(
        TrajectoryStep(
            player_id="p1",
            turn_index=0,
            observation=observation(10),
            legal_action_mask=LEGAL_TWO_ACTION_MASK,
            action_index=0,
        )
    )
    trajectory.record_terminal(TerminalState(winner=None, turn_count=250, capped=True))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "capped"},
        decision_round_count=1,
        elapsed_seconds=0.1,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def write_record(path: Path, record: RolloutRecord) -> None:
    with path.open("w", encoding="utf-8") as handle:
        write_rollout_record(handle, record)


class LinearPolicyTest(unittest.TestCase):
    def test_train_linear_policy_learns_separable_rollout_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            write_record(data_path, separable_record())

            result = train_linear_policy(
                data_path,
                config=LinearTrainingConfig(
                    feature_count=512,
                    epochs=8,
                    learning_rate=0.01,
                    window_size=1,
                ),
            )
            metrics = evaluate_linear_policy(data_path, result.model)

        self.assertLess(result.final_metrics.loss, 0.1)
        self.assertEqual(metrics.accuracy, 1.0)

    def test_linear_checkpoint_round_trip_preserves_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            checkpoint_path = Path(temp_dir) / "linear.json"
            write_record(data_path, separable_record())
            model = train_linear_policy(
                data_path,
                config=LinearTrainingConfig(feature_count=128, epochs=2, learning_rate=0.01),
            ).model
            save_linear_model(checkpoint_path, model)

            restored = load_linear_model(checkpoint_path)
            features = features_from_observation_window(
                [observation(10)],
                window_size=restored.window_size,
                feature_count=restored.feature_count,
            )

        self.assertEqual(restored.to_dict(), model.to_dict())
        self.assertEqual(restored.action_schema_version, ACTION_SCHEMA_VERSION)
        self.assertEqual(restored.observation_schema_version, OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(restored.feature_schema_version, LINEAR_FEATURE_SCHEMA_VERSION)
        self.assertEqual(restored.feature_fingerprint, linear_feature_fingerprint())
        self.assertEqual(restored.predict_action(features, LEGAL_TWO_ACTION_MASK), model.predict_action(features, LEGAL_TWO_ACTION_MASK))

    def test_linear_model_omits_default_zero_opponent_head(self) -> None:
        model = LinearPolicyModel.initialized(feature_count=8, window_size=1)
        probabilities = model.opponent_action_probabilities({0: 1.0})

        self.assertEqual(model.opponent_weights, ())
        self.assertEqual(model.to_dict()["opponent_weights"], [])
        self.assertEqual(probabilities, tuple(1.0 / ACTION_COUNT for _ in range(ACTION_COUNT)))
        self.assertEqual(model.predict_opponent_action({0: 1.0}), 0)

    def test_load_linear_model_compacts_legacy_zero_opponent_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "legacy-linear.json"
            payload = LinearPolicyModel.initialized(feature_count=8, window_size=1).to_dict()
            payload["opponent_weights"] = [[0.0 for _ in range(8)] for _ in range(ACTION_COUNT)]
            checkpoint_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            with patch.object(linear_policy_module.json, "loads", wraps=json.loads) as loads:
                restored = load_linear_model(checkpoint_path)

        self.assertIn('"opponent_weights": []', loads.call_args.args[0])
        self.assertEqual(restored.opponent_weights, ())

    def test_load_linear_model_preserves_nonzero_opponent_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "linear-with-opponent-head.json"
            payload = LinearPolicyModel.initialized(feature_count=8, window_size=1).to_dict()
            opponent_weights = [[0.0 for _ in range(8)] for _ in range(ACTION_COUNT)]
            opponent_weights[1][0] = 10.0
            payload["opponent_weights"] = opponent_weights
            checkpoint_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            restored = load_linear_model(checkpoint_path)

        self.assertEqual(len(restored.opponent_weights), ACTION_COUNT)
        self.assertEqual(restored.predict_opponent_action({0: 1.0}), 1)

    def test_linear_checkpoint_rejects_stale_runtime_schema_versions(self) -> None:
        payload = LinearPolicyModel.initialized(feature_count=8, window_size=1).to_dict()
        stale_payloads = []
        for key in ("action_schema_version", "observation_schema_version", "feature_schema_version"):
            mutated = dict(payload)
            mutated[key] = "stale"
            stale_payloads.append(mutated)
        for stale_payload in stale_payloads:
            with self.subTest(stale_payload=stale_payload):
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    LinearPolicyModel.from_dict(stale_payload)

        missing = dict(payload)
        del missing["observation_schema_version"]
        with self.assertRaisesRegex(ValueError, "missing required field"):
            LinearPolicyModel.from_dict(missing)

    def test_linear_checkpoint_rejects_stale_feature_fingerprint(self) -> None:
        payload = LinearPolicyModel.initialized(feature_count=8, window_size=1).to_dict()
        payload["feature_fingerprint"] = "stale"

        with self.assertRaisesRegex(ValueError, "Unsupported linear feature fingerprint"):
            LinearPolicyModel.from_dict(payload)

    def test_linear_checkpoint_rejects_stale_policy_schema_version(self) -> None:
        payload = LinearPolicyModel.initialized(feature_count=8, window_size=1).to_dict()
        payload["schema_version"] = "pokezero.linear_policy.v1"

        with self.assertRaisesRegex(ValueError, "Unsupported linear policy schema"):
            LinearPolicyModel.from_dict(payload)

    def test_linear_feature_fingerprint_payload_tracks_extractor_source_and_schemas(self) -> None:
        payload = _linear_feature_fingerprint_payload()

        self.assertEqual(payload["action_schema_version"], ACTION_SCHEMA_VERSION)
        self.assertEqual(payload["feature_schema_version"], LINEAR_FEATURE_SCHEMA_VERSION)
        self.assertEqual(payload["observation_schema_version"], OBSERVATION_SCHEMA_VERSION)
        self.assertIn("features_from_window", payload["sources"])
        self.assertIn("def features_from_window", payload["sources"]["features_from_window"])
        self.assertRegex(linear_feature_fingerprint(), r"^[0-9a-f]{64}$")

    def test_linear_feature_fingerprint_changes_when_extractor_source_changes(self) -> None:
        original = linear_feature_fingerprint()
        linear_feature_fingerprint.cache_clear()
        try:
            with patch.object(
                linear_policy_module,
                "_callable_fingerprint_source",
                side_effect=lambda function: f"changed:{function.__name__}",
            ):
                changed = linear_feature_fingerprint()
        finally:
            linear_feature_fingerprint.cache_clear()

        self.assertNotEqual(changed, original)

    def test_linear_feature_fingerprint_requires_source_files(self) -> None:
        linear_feature_fingerprint.cache_clear()
        try:
            with patch.object(linear_policy_module.inspect, "getsource", side_effect=OSError("missing")):
                with self.assertRaisesRegex(RuntimeError, "requires source files"):
                    linear_feature_fingerprint()
        finally:
            linear_feature_fingerprint.cache_clear()

    def test_linear_feature_extractor_golden_hash(self) -> None:
        features = features_from_window(
            categorical_ids=(((1, 2), (3, 4)), ((5, 6), (7, 8))),
            numeric_features=(((0.0, 1.5), (2.0, 0.0)), ((3.25, 0.0), (0.0, -1.0))),
            token_type_ids=((9, 10), (11, 12)),
            attention_mask=((True, False), (True, True)),
            history_mask=(False, True),
            feature_count=64,
        )
        encoded = json.dumps(sorted(features.items()), separators=(",", ":")).encode()

        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "e4b9231184308ca4bc20eea19d319507cde989c6f3da1a78e52b4689b2a7d31f",
        )

    def test_train_linear_policy_reports_held_out_validation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.jsonl"
            validation_path = Path(temp_dir) / "validation.jsonl"
            write_record(train_path, separable_record())
            write_record(validation_path, separable_record())

            result = train_linear_policy(
                train_path,
                config=LinearTrainingConfig(feature_count=128, epochs=2, learning_rate=0.01),
                validation_paths=validation_path,
            )

        self.assertIsNotNone(result.validation_metrics)
        assert result.validation_metrics is not None
        self.assertEqual(result.validation_metrics.examples, 24)

    def test_train_linear_policy_learns_opponent_action_auxiliary_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            write_record(data_path, opponent_action_record())

            result = train_linear_policy(
                data_path,
                config=LinearTrainingConfig(
                    feature_count=512,
                    epochs=8,
                    learning_rate=0.05,
                    opponent_action_loss_weight=1.0,
                    shuffle_buffer_size=0,
                ),
            )
            metrics = evaluate_linear_policy(data_path, result.model)
            features_for_zero = features_from_observation_window(
                [observation(10)],
                window_size=result.model.window_size,
                feature_count=result.model.feature_count,
            )
            features_for_one = features_from_observation_window(
                [observation(30)],
                window_size=result.model.window_size,
                feature_count=result.model.feature_count,
            )

        self.assertEqual(result.final_metrics.opponent_examples, 24)
        self.assertGreater(result.final_metrics.opponent_accuracy, 0.9)
        self.assertEqual(metrics.opponent_examples, 24)
        self.assertEqual(metrics.opponent_accuracy, 1.0)
        self.assertEqual(result.model.predict_opponent_action(features_for_zero), 0)
        self.assertEqual(result.model.predict_opponent_action(features_for_one), 1)

    def test_reward_weighted_objective_ignores_losing_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            write_record(data_path, losing_action_record())

            model = train_linear_policy(
                data_path,
                config=LinearTrainingConfig(
                    feature_count=128,
                    epochs=1,
                    learning_rate=0.01,
                    objective="reward-weighted",
                    shuffle_buffer_size=0,
                ),
            ).model

        self.assertEqual(model.to_dict(), LinearPolicyModel.initialized(feature_count=128, window_size=1).to_dict())

    def test_reward_weighted_objective_penalizes_capped_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            write_record(data_path, capped_action_record())

            model = train_linear_policy(
                data_path,
                config=LinearTrainingConfig(
                    feature_count=128,
                    epochs=1,
                    learning_rate=0.1,
                    capped_terminal_value=-0.25,
                    objective="reward-weighted",
                    shuffle_buffer_size=0,
                ),
            ).model
            features = features_from_observation_window(
                [observation(10)],
                window_size=model.window_size,
                feature_count=model.feature_count,
            )

        probabilities = model.action_probabilities(features, LEGAL_TWO_ACTION_MASK)
        self.assertLess(probabilities[0], probabilities[1])

    def test_reward_weighted_objective_uses_terminal_return_when_step_reward_is_zero(self) -> None:
        record = winning_action_record()
        self.assertEqual(record.trajectory.steps[0].reward, 0.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            write_record(data_path, record)

            model = train_linear_policy(
                data_path,
                config=LinearTrainingConfig(
                    feature_count=128,
                    epochs=1,
                    learning_rate=0.01,
                    objective="reward-weighted",
                    shuffle_buffer_size=0,
                ),
            ).model
            features = features_from_observation_window(
                [observation(10)],
                window_size=model.window_size,
                feature_count=model.feature_count,
            )

        self.assertEqual(model.predict_action(features, LEGAL_TWO_ACTION_MASK), 1)

    def test_shuffle_seed_keeps_training_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            write_record(data_path, separable_record())
            config = LinearTrainingConfig(
                feature_count=128,
                epochs=2,
                learning_rate=0.01,
                shuffle_buffer_size=4,
                shuffle_seed=99,
            )

            first = train_linear_policy(data_path, config=config).model
            second = train_linear_policy(data_path, config=config).model

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_train_linear_policy_can_warm_start_from_initial_model(self) -> None:
        weights = [[0.0 for _ in range(128)] for _ in range(9)]
        weights[1][0] = 2.0
        initial_model = LinearPolicyModel(
            policy_id="warm-start",
            feature_count=128,
            window_size=1,
            weights=tuple(tuple(row) for row in weights),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            write_record(data_path, winning_action_record())

            model = train_linear_policy(
                data_path,
                config=LinearTrainingConfig(
                    feature_count=128,
                    epochs=1,
                    learning_rate=0.01,
                    objective="reward-weighted",
                    shuffle_buffer_size=0,
                    policy_id="warm-started",
                ),
                initial_model=initial_model,
            ).model

        self.assertEqual(model.policy_id, "warm-started")
        self.assertGreater(model.weights[1][0], 2.0)

    def test_train_linear_policy_rejects_incompatible_initial_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            write_record(data_path, winning_action_record())

            with self.assertRaisesRegex(ValueError, "feature_count"):
                train_linear_policy(
                    data_path,
                    config=LinearTrainingConfig(feature_count=128),
                    initial_model=LinearPolicyModel.initialized(feature_count=64, window_size=1),
                )

    def test_linear_softmax_policy_respects_legal_mask(self) -> None:
        weights = [[0.0 for _ in range(8)] for _ in range(9)]
        weights[1][0] = 100.0
        model = LinearPolicyModel(
            policy_id="linear-test",
            feature_count=8,
            window_size=1,
            weights=tuple(tuple(row) for row in weights),
        )
        policy = LinearSoftmaxPolicy(model=model)
        obs = PokeZeroObservationV0(
            categorical_ids=observation(10).categorical_ids,
            numeric_features=observation(10).numeric_features,
            token_type_ids=observation(10).token_type_ids,
            attention_mask=observation(10).attention_mask,
            legal_action_mask=(True, False, False, False, False, False, False, False, False),
            perspective=ObservationPerspective.from_showdown_slot("p1", "p1"),
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.policy_id, "linear-test")
        self.assertEqual(decision.action_probability, 1.0)

    def test_linear_softmax_policy_samples_and_records_sampling_probability(self) -> None:
        model = LinearPolicyModel.initialized(
            feature_count=8,
            window_size=1,
            policy_id="linear-test",
        )
        policy = LinearSoftmaxPolicy(model=model, deterministic=False)
        obs = PokeZeroObservationV0(
            categorical_ids=observation(10).categorical_ids,
            numeric_features=observation(10).numeric_features,
            token_type_ids=observation(10).token_type_ids,
            attention_mask=observation(10).attention_mask,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            perspective=ObservationPerspective.from_showdown_slot("p1", "p1"),
        )

        decision = policy.select_action(obs, rng=random.Random(1))

        self.assertIn(decision.action_index, {0, 1})
        self.assertAlmostEqual(decision.action_probability, 0.5)
        self.assertFalse(decision.metadata["deterministic"])

    def test_linear_softmax_policy_records_epsilon_mixture_probability(self) -> None:
        model = LinearPolicyModel.initialized(
            feature_count=8,
            window_size=1,
            policy_id="linear-test",
        )
        policy = LinearSoftmaxPolicy(model=model, deterministic=True, exploration_epsilon=0.2)
        obs = PokeZeroObservationV0(
            categorical_ids=observation(10).categorical_ids,
            numeric_features=observation(10).numeric_features,
            token_type_ids=observation(10).token_type_ids,
            attention_mask=observation(10).attention_mask,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            perspective=ObservationPerspective.from_showdown_slot("p1", "p1"),
        )

        decision = policy.select_action(obs, rng=random.Random(2))

        self.assertEqual(decision.action_index, 0)
        self.assertAlmostEqual(decision.action_probability, 0.9)

    def test_linear_softmax_policy_records_non_greedy_epsilon_probability(self) -> None:
        model = LinearPolicyModel.initialized(
            feature_count=8,
            window_size=1,
            policy_id="linear-test",
        )
        policy = LinearSoftmaxPolicy(model=model, deterministic=True, exploration_epsilon=0.2)
        obs = PokeZeroObservationV0(
            categorical_ids=observation(10).categorical_ids,
            numeric_features=observation(10).numeric_features,
            token_type_ids=observation(10).token_type_ids,
            attention_mask=observation(10).attention_mask,
            legal_action_mask=(True, True, False, False, False, False, False, False, False),
            perspective=ObservationPerspective.from_showdown_slot("p1", "p1"),
        )

        decision = policy.select_action(obs, rng=random.Random(18))

        self.assertEqual(decision.action_index, 1)
        self.assertAlmostEqual(decision.action_probability, 0.1)

    def test_linear_cli_train_and_evaluate_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "rollouts.jsonl"
            checkpoint_path = Path(temp_dir) / "linear.json"
            write_record(data_path, separable_record())

            with patch("sys.stdout"):
                train_exit = linear_cli_main(
                    [
                        "train",
                        "--data",
                        str(data_path),
                        "--validation-data",
                        str(data_path),
                        "--out",
                        str(checkpoint_path),
                        "--epochs",
                        "2",
                        "--learning-rate",
                        "0.01",
                        "--feature-count",
                        "128",
                        "--objective",
                        "reward-weighted",
                        "--shuffle-buffer-size",
                        "4",
                    ]
                )
                evaluate_exit = linear_cli_main(
                    [
                        "evaluate",
                        "--data",
                        str(data_path),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--max-examples",
                        "4",
                    ]
                )

        self.assertEqual(train_exit, 0)
        self.assertEqual(evaluate_exit, 0)


if __name__ == "__main__":
    unittest.main()
