from pathlib import Path
import tempfile
import unittest

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.dataset import (
    MISSING_ACTION_INDEX,
    TrajectoryDatasetConfig,
    batch_training_examples,
    examples_from_record,
    iter_training_batches,
    iter_training_examples,
    training_batch_from_examples,
)
from pokezero.env import TerminalState
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


MASK = (True, False, False, False, False, False, False, False, False)


def observation(value: int) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((value,) for _ in range(spec.token_count)),
        numeric_features=tuple((float(value),) for _ in range(spec.token_count)),
        token_type_ids=tuple(value for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=MASK,
    )


def step(
    *,
    player_id: str,
    turn_index: int,
    value: int,
    reward: float,
    opponent_action_index: int | None = None,
    action_probability: float | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        player_id=player_id,
        turn_index=turn_index,
        observation=observation(value),
        legal_action_mask=MASK,
        action_index=0,
        reward=reward,
        opponent_action_index=opponent_action_index,
        action_probability=action_probability,
        metadata={"value": value},
    )


def rollout_record() -> RolloutRecord:
    trajectory = BattleTrajectory(
        battle_id="battle-1",
        format_id="gen3randombattle",
        seed=123,
    )
    trajectory.append(
        step(
            player_id="p1",
            turn_index=0,
            value=5,
            reward=1.0,
            opponent_action_index=1,
            action_probability=0.5,
        )
    )
    trajectory.append(step(player_id="p2", turn_index=0, value=50, reward=-1.0))
    trajectory.append(step(player_id="p1", turn_index=1, value=6, reward=3.0))
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=2))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "test", "p2": "test"},
        decision_round_count=2,
        elapsed_seconds=0.1,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


class DatasetTest(unittest.TestCase):
    def test_examples_use_same_player_history_windows_and_discounted_returns(self) -> None:
        examples = list(
            examples_from_record(
                rollout_record(),
                config=TrajectoryDatasetConfig(window_size=2, discount=0.5),
            )
        )

        p1_first = examples[0]
        p2_first = examples[1]
        p1_second = examples[2]

        self.assertEqual(p1_first.history_mask, (False, True))
        self.assertEqual(p1_first.categorical_ids[0][0][0], 0)
        self.assertEqual(p1_first.categorical_ids[1][0][0], 5)
        self.assertEqual(p2_first.history_mask, (False, True))
        self.assertEqual(p1_second.history_mask, (True, True))
        self.assertEqual(p1_second.categorical_ids[0][0][0], 5)
        self.assertEqual(p1_second.categorical_ids[1][0][0], 6)
        self.assertAlmostEqual(p1_first.return_value, 2.5)
        self.assertAlmostEqual(p1_second.return_value, 3.0)
        self.assertAlmostEqual(p2_first.return_value, -1.0)

    def test_training_batch_preserves_labels_and_optional_field_masks(self) -> None:
        examples = list(examples_from_record(rollout_record(), config=TrajectoryDatasetConfig(window_size=2)))

        batch = training_batch_from_examples(examples[:2])

        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.window_size, 2)
        self.assertEqual(batch.action_indices, (0, 0))
        self.assertEqual(batch.legal_action_mask, (MASK, MASK))
        self.assertEqual(batch.opponent_action_indices, (1, MISSING_ACTION_INDEX))
        self.assertEqual(batch.opponent_action_mask, (True, False))
        self.assertEqual(batch.action_probabilities, (0.5, 0.0))
        self.assertEqual(batch.action_probability_mask, (True, False))
        self.assertEqual(batch.battle_ids, ("battle-1", "battle-1"))
        self.assertEqual(batch.step_metadata[0]["value"], 5)

    def test_batch_training_examples_chunks_stream_and_keeps_tail_batch(self) -> None:
        examples = list(examples_from_record(rollout_record(), config=TrajectoryDatasetConfig(window_size=1)))

        batches = list(batch_training_examples(examples, batch_size=2))

        self.assertEqual([batch.batch_size for batch in batches], [2, 1])

    def test_iter_training_examples_and_batches_stream_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())

            examples = list(iter_training_examples(path, config=TrajectoryDatasetConfig(window_size=1)))
            batches = list(iter_training_batches(path, batch_size=2, config=TrajectoryDatasetConfig(window_size=1)))

        self.assertEqual(len(examples), 3)
        self.assertEqual([batch.batch_size for batch in batches], [2, 1])

    def test_dataset_config_validates_window_and_discount(self) -> None:
        with self.assertRaisesRegex(ValueError, "window_size"):
            TrajectoryDatasetConfig(window_size=0)
        with self.assertRaisesRegex(ValueError, "discount"):
            TrajectoryDatasetConfig(discount=1.5)

    def test_training_batch_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            training_batch_from_examples([])


if __name__ == "__main__":
    unittest.main()
