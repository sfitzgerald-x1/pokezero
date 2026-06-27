from dataclasses import replace
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


def observation(value: int, *, metadata: dict | None = None) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((value,) for _ in range(spec.token_count)),
        numeric_features=tuple((float(value),) for _ in range(spec.token_count)),
        token_type_ids=tuple(value for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=MASK,
        metadata=metadata or {},
    )


def step(
    *,
    player_id: str,
    turn_index: int,
    value: int,
    reward: float,
    opponent_action_index: int | None = None,
    action_probability: float | None = None,
    value_estimate: float | None = None,
    observation_metadata: dict | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        player_id=player_id,
        turn_index=turn_index,
        observation=observation(value, metadata=observation_metadata),
        legal_action_mask=MASK,
        action_index=0,
        reward=reward,
        opponent_action_index=opponent_action_index,
        action_probability=action_probability,
        value_estimate=value_estimate,
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
    def test_examples_use_same_player_history_windows_and_terminal_returns(self) -> None:
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
        self.assertAlmostEqual(p1_first.return_value, 0.5)
        self.assertAlmostEqual(p1_second.return_value, 1.0)
        self.assertAlmostEqual(p2_first.return_value, -1.0)

    def test_returns_use_terminal_winner_even_when_winner_has_no_final_step_reward(self) -> None:
        trajectory = BattleTrajectory(battle_id="asymmetric", format_id="gen3randombattle", seed=5)
        trajectory.append(step(player_id="p1", turn_index=0, value=5, reward=0.0))
        trajectory.append(step(player_id="p2", turn_index=1, value=50, reward=-1.0))
        trajectory.record_terminal(TerminalState(winner="p1", turn_count=2))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "test", "p2": "test"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )
        p1_examples = [
            example
            for example in examples_from_record(record, config=TrajectoryDatasetConfig(window_size=1))
            if example.player_id == "p1"
        ]

        self.assertEqual([example.reward for example in p1_examples], [0.0])
        self.assertEqual([example.return_value for example in p1_examples], [1.0])

    def test_capped_or_tied_terminal_returns_default_to_zero(self) -> None:
        record = rollout_record()
        terminal = TerminalState(winner=None, turn_count=250, capped=True)
        record.trajectory.record_terminal(terminal)
        record = replace(record, terminal=terminal)

        examples = list(examples_from_record(record, config=TrajectoryDatasetConfig(window_size=1)))

        self.assertEqual({example.return_value for example in examples}, {0.0})

    def test_capped_terminal_value_can_penalize_both_players(self) -> None:
        record = rollout_record()
        terminal = TerminalState(winner=None, turn_count=250, capped=True)
        record.trajectory.record_terminal(terminal)
        record = replace(record, terminal=terminal)

        examples = list(
            examples_from_record(
                record,
                config=TrajectoryDatasetConfig(window_size=1, capped_terminal_value=-0.25),
            )
        )

        self.assertEqual({example.return_value for example in examples}, {-0.25})
        self.assertTrue(all(example.terminal_capped for example in examples))

    def test_hp_delta_return_shaping_uses_visible_player_relative_changes(self) -> None:
        trajectory = BattleTrajectory(battle_id="hp-shaping", format_id="gen3randombattle", seed=5)
        trajectory.append(
            step(
                player_id="p1",
                turn_index=0,
                value=1,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 1.0}],
                    "opponent_team": [{"species": "Xatu", "hp_fraction": 1.0}],
                },
            )
        )
        trajectory.append(
            step(
                player_id="p1",
                turn_index=1,
                value=2,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 1.0}],
                    "opponent_team": [{"species": "Xatu", "hp_fraction": 0.4}],
                },
            )
        )
        trajectory.record_terminal(TerminalState(winner=None, turn_count=2))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "test"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )

        examples = list(
            examples_from_record(
                record,
                config=TrajectoryDatasetConfig(window_size=1, hp_delta_return_weight=3.0),
            )
        )

        self.assertAlmostEqual(examples[0].return_value, 0.3)
        self.assertAlmostEqual(examples[1].return_value, 0.3)

    def test_return_shaping_clips_targets_to_bounded_value_range(self) -> None:
        trajectory = BattleTrajectory(battle_id="clipped-shaping", format_id="gen3randombattle", seed=5)
        trajectory.append(
            step(
                player_id="p1",
                turn_index=0,
                value=1,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 1.0}],
                    "opponent_team": [{"species": "Xatu", "hp_fraction": 1.0}],
                },
            )
        )
        trajectory.append(
            step(
                player_id="p1",
                turn_index=1,
                value=2,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 1.0}],
                    "opponent_team": [{"species": "Xatu", "hp_fraction": 0.0}],
                },
            )
        )
        trajectory.record_terminal(TerminalState(winner=None, turn_count=2))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "test"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )

        examples = list(
            examples_from_record(
                record,
                config=TrajectoryDatasetConfig(window_size=1, hp_delta_return_weight=12.0),
            )
        )

        self.assertAlmostEqual(examples[0].return_value, 1.0)
        self.assertAlmostEqual(examples[1].return_value, 1.0)

    def test_faint_delta_return_shaping_rewards_new_visible_opponent_faints(self) -> None:
        trajectory = BattleTrajectory(battle_id="faint-shaping", format_id="gen3randombattle", seed=5)
        trajectory.append(
            step(
                player_id="p1",
                turn_index=0,
                value=1,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "fainted": False}],
                    "opponent_team": [{"species": "Xatu", "fainted": False}],
                },
            )
        )
        trajectory.append(
            step(
                player_id="p1",
                turn_index=1,
                value=2,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "fainted": False}],
                    "opponent_team": [{"species": "Xatu", "fainted": True}],
                },
            )
        )
        trajectory.record_terminal(TerminalState(winner=None, turn_count=2))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "test"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )

        examples = list(
            examples_from_record(
                record,
                config=TrajectoryDatasetConfig(window_size=1, faint_delta_return_weight=1.2),
            )
        )

        self.assertAlmostEqual(examples[0].return_value, 0.2)
        self.assertAlmostEqual(examples[1].return_value, 0.2)

    def test_return_shaping_penalizes_visible_self_side_damage_and_faints(self) -> None:
        trajectory = BattleTrajectory(battle_id="self-damage-shaping", format_id="gen3randombattle", seed=5)
        trajectory.append(
            step(
                player_id="p1",
                turn_index=0,
                value=1,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 1.0, "fainted": False}],
                    "opponent_team": [{"species": "Xatu", "hp_fraction": 1.0, "fainted": False}],
                },
            )
        )
        trajectory.append(
            step(
                player_id="p1",
                turn_index=1,
                value=2,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 0.4, "fainted": True}],
                    "opponent_team": [{"species": "Xatu", "hp_fraction": 1.0, "fainted": False}],
                },
            )
        )
        trajectory.record_terminal(TerminalState(winner=None, turn_count=2))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "test"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )

        examples = list(
            examples_from_record(
                record,
                config=TrajectoryDatasetConfig(
                    window_size=1,
                    hp_delta_return_weight=3.0,
                    faint_delta_return_weight=1.2,
                ),
            )
        )

        self.assertAlmostEqual(examples[0].return_value, -0.5)
        self.assertAlmostEqual(examples[1].return_value, -0.5)

    def test_return_shaping_discounts_future_shaping_rewards(self) -> None:
        trajectory = BattleTrajectory(battle_id="discount-shaping", format_id="gen3randombattle", seed=5)
        trajectory.append(
            step(
                player_id="p1",
                turn_index=0,
                value=1,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 1.0}],
                    "opponent_team": [{"species": "Xatu", "hp_fraction": 1.0}],
                },
            )
        )
        trajectory.append(
            step(
                player_id="p1",
                turn_index=1,
                value=2,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 1.0}],
                    "opponent_team": [{"species": "Xatu", "hp_fraction": 0.4}],
                },
            )
        )
        trajectory.record_terminal(TerminalState(winner=None, turn_count=2))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "test"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )

        examples = list(
            examples_from_record(
                record,
                config=TrajectoryDatasetConfig(window_size=1, discount=0.5, hp_delta_return_weight=6.0),
            )
        )

        self.assertAlmostEqual(examples[0].return_value, 0.3)
        self.assertAlmostEqual(examples[1].return_value, 0.6)

    def test_return_shaping_ignores_newly_revealed_opponent_without_prior_baseline(self) -> None:
        trajectory = BattleTrajectory(battle_id="reveal-shaping", format_id="gen3randombattle", seed=5)
        trajectory.append(
            step(
                player_id="p1",
                turn_index=0,
                value=1,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 1.0}],
                    "opponent_team": [{"species": "Xatu", "hp_fraction": 1.0}],
                },
            )
        )
        trajectory.append(
            step(
                player_id="p1",
                turn_index=1,
                value=2,
                reward=0.0,
                observation_metadata={
                    "self_team": [{"species": "Charizard", "hp_fraction": 1.0}],
                    "opponent_team": [
                        {"species": "Xatu", "hp_fraction": 1.0},
                        {"species": "Tauros", "hp_fraction": 0.4, "fainted": True},
                    ],
                },
            )
        )
        trajectory.record_terminal(TerminalState(winner=None, turn_count=2))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "test"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )

        examples = list(
            examples_from_record(
                record,
                config=TrajectoryDatasetConfig(
                    window_size=1,
                    hp_delta_return_weight=10.0,
                    faint_delta_return_weight=10.0,
                ),
            )
        )

        self.assertEqual([example.return_value for example in examples], [0.0, 0.0])

    def test_turn_penalty_return_shaping_applies_after_threshold(self) -> None:
        record = rollout_record()
        terminal = TerminalState(winner=None, turn_count=250, capped=True)
        record.trajectory.record_terminal(terminal)
        record = replace(record, terminal=terminal)

        examples = list(
            examples_from_record(
                record,
                config=TrajectoryDatasetConfig(window_size=1, turn_penalty_after=1, turn_penalty=0.2),
            )
        )

        self.assertAlmostEqual(examples[0].return_value, -0.2)
        self.assertAlmostEqual(examples[1].return_value, 0.0)
        self.assertAlmostEqual(examples[2].return_value, -0.2)

    def test_gae_ppo_targets_use_recorded_behavior_value_estimates(self) -> None:
        trajectory = BattleTrajectory(battle_id="gae", format_id="gen3randombattle", seed=5)
        trajectory.append(step(player_id="p1", turn_index=0, value=5, reward=0.0, value_estimate=0.2))
        trajectory.append(step(player_id="p2", turn_index=0, value=50, reward=0.0))
        trajectory.append(step(player_id="p1", turn_index=1, value=6, reward=0.0, value_estimate=0.5))
        trajectory.record_terminal(TerminalState(winner="p1", turn_count=2))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "neural", "p2": "fixed"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )

        examples = list(
            examples_from_record(
                record,
                config=TrajectoryDatasetConfig(window_size=1, ppo_target_mode="gae", gae_lambda=1.0),
            )
        )

        self.assertAlmostEqual(examples[0].return_value, 1.0)
        self.assertAlmostEqual(examples[0].value_estimate, 0.2)
        self.assertAlmostEqual(examples[0].ppo_advantage, 0.8)
        self.assertAlmostEqual(examples[0].ppo_value_target, 1.0)
        self.assertIsNone(examples[1].ppo_advantage)
        self.assertIsNone(examples[1].ppo_value_target)
        self.assertAlmostEqual(examples[2].ppo_advantage, 0.5)
        self.assertAlmostEqual(examples[2].ppo_value_target, 1.0)

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
        self.assertEqual(batch.value_estimates, (0.0, 0.0))
        self.assertEqual(batch.value_estimate_mask, (False, False))
        self.assertEqual(batch.ppo_advantages, (0.0, 0.0))
        self.assertEqual(batch.ppo_advantage_mask, (False, False))
        self.assertEqual(batch.ppo_value_targets, (0.0, 0.0))
        self.assertEqual(batch.ppo_value_target_mask, (False, False))
        self.assertEqual(batch.battle_ids, ("battle-1", "battle-1"))
        self.assertEqual(batch.terminal_capped, (False, False))
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
        with self.assertRaisesRegex(ValueError, "capped_terminal_value"):
            TrajectoryDatasetConfig(capped_terminal_value=0.5)
        with self.assertRaisesRegex(ValueError, "hp_delta_return_weight"):
            TrajectoryDatasetConfig(hp_delta_return_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "faint_delta_return_weight"):
            TrajectoryDatasetConfig(faint_delta_return_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "turn_penalty_after"):
            TrajectoryDatasetConfig(turn_penalty_after=-1)
        with self.assertRaisesRegex(ValueError, "turn_penalty"):
            TrajectoryDatasetConfig(turn_penalty=-0.1)
        with self.assertRaisesRegex(ValueError, "turn_penalty_after"):
            TrajectoryDatasetConfig(turn_penalty=0.1)
        with self.assertRaisesRegex(ValueError, "ppo_target_mode"):
            TrajectoryDatasetConfig(ppo_target_mode="bad")
        with self.assertRaisesRegex(ValueError, "gae_lambda"):
            TrajectoryDatasetConfig(gae_lambda=1.5)

    def test_training_batch_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            training_batch_from_examples([])


if __name__ == "__main__":
    unittest.main()
