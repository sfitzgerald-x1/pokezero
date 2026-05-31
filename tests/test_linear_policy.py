from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.env import TerminalState
from pokezero.linear_cli import main as linear_cli_main
from pokezero.linear_policy import (
    LinearPolicyModel,
    LinearSoftmaxPolicy,
    LinearTrainingConfig,
    evaluate_linear_policy,
    features_from_observation_window,
    load_linear_model,
    save_linear_model,
    train_linear_policy,
)
from pokezero.observation import ObservationPerspective, ObservationSpec, PokeZeroObservationV0
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
        self.assertEqual(restored.predict_action(features, LEGAL_TWO_ACTION_MASK), model.predict_action(features, LEGAL_TWO_ACTION_MASK))

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

    def test_reward_weighted_objective_reinforces_winning_actions(self) -> None:
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
