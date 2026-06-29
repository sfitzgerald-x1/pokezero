from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.dataset import (
    MISSING_ACTION_INDEX,
    TRAINING_CACHE_SCHEMA_VERSION,
    TrajectoryDatasetConfig,
    TrainingCacheBuilder,
    batch_training_examples,
    delete_training_cache_path,
    examples_from_record,
    is_training_cache_path,
    iter_training_cache_batches,
    iter_training_batches,
    iter_training_examples,
    training_cache_root_byte_size,
    training_cache_paths_byte_size,
    training_batch_from_examples,
    write_training_cache_from_rollouts,
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
            value_estimate=0.125,
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

    def test_gae_ppo_targets_fall_back_for_player_with_dropped_value_estimate(self) -> None:
        trajectory = BattleTrajectory(battle_id="gae", format_id="gen3randombattle", seed=5)
        trajectory.append(step(player_id="p1", turn_index=0, value=5, reward=0.0, value_estimate=0.2))
        trajectory.append(step(player_id="p1", turn_index=1, value=6, reward=0.0))
        trajectory.record_terminal(TerminalState(winner="p1", turn_count=2))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "neural"},
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
        self.assertAlmostEqual(examples[1].return_value, 1.0)
        self.assertIsNone(examples[0].ppo_advantage)
        self.assertIsNone(examples[0].ppo_value_target)
        self.assertIsNone(examples[1].ppo_advantage)
        self.assertIsNone(examples[1].ppo_value_target)

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
        self.assertEqual(batch.value_estimates, (0.125, 0.0))
        self.assertEqual(batch.value_estimate_mask, (True, False))
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

    def test_training_cache_round_trips_raw_training_batches(self) -> None:
        self._require_numpy()
        config = TrajectoryDatasetConfig(window_size=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            cache_path = Path(temp_dir) / "cache"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())

            summary = write_training_cache_from_rollouts(path, cache_path, config=config)
            raw_batches = list(iter_training_batches(path, batch_size=2, config=config))
            cached_batches = list(iter_training_batches(cache_path, batch_size=2, config=config))
            explicit_cached_batches = list(iter_training_cache_batches(cache_path, batch_size=2, config=config))

            self.assertTrue(is_training_cache_path(summary.path))
            self.assertEqual(summary.record_count, 1)
            self.assertEqual(summary.example_count, 3)
            self.assertGreater(summary.byte_size, 0)
            self.assertEqual([batch.batch_size for batch in cached_batches], [2, 1])
            self.assertEqual(_batch_payload(cached_batches), _batch_payload(raw_batches))
            self.assertEqual(_batch_payload(explicit_cached_batches), _batch_payload(raw_batches))

    def test_training_cache_batches_coalesce_across_cache_paths(self) -> None:
        self._require_numpy()
        config = TrajectoryDatasetConfig(window_size=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_paths = tuple(temp_path / f"rollouts-{index}.jsonl" for index in range(3))
            cache_paths = tuple(temp_path / f"cache-{index}" for index in range(3))
            for path in jsonl_paths:
                with path.open("w", encoding="utf-8") as handle:
                    write_rollout_record(handle, rollout_record())
            for jsonl_path, cache_path in zip(jsonl_paths, cache_paths, strict=True):
                write_training_cache_from_rollouts(jsonl_path, cache_path, config=config)

            consumed: list[Path] = []
            raw_batches = list(iter_training_batches(jsonl_paths, batch_size=4, config=config))
            cached_batches = list(
                iter_training_batches(
                    cache_paths,
                    batch_size=4,
                    config=config,
                    consumed_cache_callback=consumed.append,
                )
            )

        self.assertEqual([batch.batch_size for batch in cached_batches], [4, 4, 1])
        self.assertEqual(_batch_payload(cached_batches), _batch_payload(raw_batches))
        self.assertEqual([batch.battle_ids for batch in cached_batches], [("", "", "", ""), ("", "", "", ""), ("",)])
        self.assertEqual([batch.step_metadata for batch in cached_batches], [({}, {}, {}, {}), ({}, {}, {}, {}), ({},)])
        self.assertEqual(consumed, list(cache_paths))

    def test_deferred_training_cache_batches_trim_sliced_row_tables(self) -> None:
        self._require_numpy()
        config = TrajectoryDatasetConfig(window_size=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_paths = tuple(temp_path / f"rollouts-{index}.jsonl" for index in range(2))
            cache_paths = tuple(temp_path / f"cache-{index}" for index in range(2))
            for path in jsonl_paths:
                with path.open("w", encoding="utf-8") as handle:
                    write_rollout_record(handle, rollout_record())
            for jsonl_path, cache_path in zip(jsonl_paths, cache_paths, strict=True):
                write_training_cache_from_rollouts(jsonl_path, cache_path, config=config)

            cached_batches = list(
                iter_training_batches(
                    cache_paths,
                    batch_size=4,
                    config=config,
                    defer_cache_window_expansion=True,
                )
            )

        self.assertEqual([batch.batch_size for batch in cached_batches], [4, 2])
        self.assertEqual([batch.row_categorical_ids.shape[0] for batch in cached_batches], [4, 2])
        self.assertEqual(_tolist(cached_batches[0].window_row_indices), [[0], [1], [2], [3]])
        self.assertEqual(_tolist(cached_batches[1].window_row_indices), [[0], [1]])
        self.assertEqual(_tolist(cached_batches[0].row_categorical_ids[:, 0, 0]), [5, 50, 6, 5])
        self.assertEqual(_tolist(cached_batches[1].row_categorical_ids[:, 0, 0]), [50, 6])

    def test_training_cache_round_trips_gae_ppo_training_targets(self) -> None:
        self._require_numpy()
        trajectory = BattleTrajectory(battle_id="gae-cache", format_id="gen3randombattle", seed=5)
        trajectory.append(step(player_id="p1", turn_index=0, value=5, reward=0.0, value_estimate=0.25))
        trajectory.append(step(player_id="p2", turn_index=0, value=50, reward=0.0, value_estimate=-0.25))
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
        config = TrajectoryDatasetConfig(window_size=1, ppo_target_mode="gae", gae_lambda=1.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            cache_path = Path(temp_dir) / "cache"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, record)

            write_training_cache_from_rollouts(path, cache_path, config=config)
            raw_batches = list(iter_training_batches(path, batch_size=2, config=config))
            cached_batches = list(iter_training_batches(cache_path, batch_size=2, config=config))

            self.assertEqual(_batch_payload(cached_batches), _batch_payload(raw_batches))

    def test_training_cache_reader_defaults_missing_value_estimate_arrays(self) -> None:
        self._require_numpy()
        builder = TrainingCacheBuilder(config=TrajectoryDatasetConfig(window_size=1))
        builder.add_record(rollout_record())
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            builder.write(cache_path)
            (cache_path / "value_estimates.npy").unlink()
            (cache_path / "value_estimate_mask.npy").unlink()

            batches = list(iter_training_cache_batches(cache_path, batch_size=2))

        self.assertEqual(_tolist(batches[0].value_estimates), [0.0, 0.0])
        self.assertEqual(_tolist(batches[0].value_estimate_mask), [False, False])

    def test_training_cache_reader_reports_missing_required_array(self) -> None:
        self._require_numpy()
        builder = TrainingCacheBuilder(config=TrajectoryDatasetConfig(window_size=1))
        builder.add_record(rollout_record())
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            builder.write(cache_path)
            (cache_path / "returns.npy").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "returns"):
                list(iter_training_cache_batches(cache_path, batch_size=2))

    def test_training_cache_builder_writes_schema_metadata(self) -> None:
        self._require_numpy()
        builder = TrainingCacheBuilder(config=TrajectoryDatasetConfig(window_size=1, discount=0.5))
        builder.add_record(rollout_record())
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            builder.write(cache_path)
            metadata = (cache_path / "metadata.json").read_text(encoding="utf-8")

        self.assertIn(TRAINING_CACHE_SCHEMA_VERSION, metadata)
        self.assertIn('"discount": 0.5', metadata)

    def test_training_cache_compacts_zero_padded_categorical_features(self) -> None:
        self._require_numpy()

        spec = ObservationSpec(categorical_feature_count=4, numeric_feature_count=1)

        def wide_observation(value: int) -> PokeZeroObservationV0:
            return PokeZeroObservationV0(
                categorical_ids=tuple((0, value, 0, value + 1) for _ in range(spec.token_count)),
                numeric_features=tuple((float(value),) for _ in range(spec.token_count)),
                token_type_ids=tuple(0 for _ in range(spec.token_count)),
                attention_mask=tuple(True for _ in range(spec.token_count)),
                legal_action_mask=MASK,
            )

        trajectory = BattleTrajectory(battle_id="compact-cache", format_id="gen3randombattle", seed=7)
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=wide_observation(5),
                legal_action_mask=MASK,
                action_index=0,
                reward=0.0,
            )
        )
        trajectory.record_terminal(TerminalState(winner="p1", turn_count=1))
        record = RolloutRecord(
            battle_id=trajectory.battle_id,
            seed=trajectory.seed,
            format_id=trajectory.format_id,
            policy_ids={"p1": "test"},
            decision_round_count=1,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            cache_path = Path(temp_dir) / "cache"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, record)

            write_training_cache_from_rollouts(path, cache_path, config=TrajectoryDatasetConfig(window_size=1))
            metadata = json.loads((cache_path / "metadata.json").read_text(encoding="utf-8"))
            cached_batch = next(iter_training_batches(cache_path, batch_size=1, config=TrajectoryDatasetConfig(window_size=1)))

        self.assertEqual(metadata["categorical_storage"]["mode"], "compact-nonzero")
        self.assertEqual(metadata["categorical_storage"]["original_feature_count"], 4)
        self.assertEqual(metadata["categorical_storage"]["stored_feature_count"], 2)
        self.assertEqual(cached_batch.categorical_ids.shape[-1], 2)
        self.assertEqual(_tolist(cached_batch.categorical_ids)[0][0][0], [5, 6])

    def test_training_cache_coalesces_mixed_categorical_widths(self) -> None:
        self._require_numpy()

        spec = ObservationSpec(categorical_feature_count=4, numeric_feature_count=1)

        def record_with_categories(battle_id: str, categories: tuple[int, int, int, int]) -> RolloutRecord:
            observation_payload = PokeZeroObservationV0(
                categorical_ids=tuple(categories for _ in range(spec.token_count)),
                numeric_features=tuple((1.0,) for _ in range(spec.token_count)),
                token_type_ids=tuple(0 for _ in range(spec.token_count)),
                attention_mask=tuple(True for _ in range(spec.token_count)),
                legal_action_mask=MASK,
            )
            trajectory = BattleTrajectory(battle_id=battle_id, format_id="gen3randombattle", seed=7)
            trajectory.append(
                TrajectoryStep(
                    player_id="p1",
                    turn_index=0,
                    observation=observation_payload,
                    legal_action_mask=MASK,
                    action_index=0,
                    reward=0.0,
                )
            )
            trajectory.record_terminal(TerminalState(winner="p1", turn_count=1))
            return RolloutRecord(
                battle_id=trajectory.battle_id,
                seed=trajectory.seed,
                format_id=trajectory.format_id,
                policy_ids={"p1": "test"},
                decision_round_count=1,
                elapsed_seconds=0.1,
                terminal=trajectory.terminal,
                trajectory=trajectory,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dense_jsonl = temp_path / "dense.jsonl"
            compact_jsonl = temp_path / "compact.jsonl"
            dense_cache = temp_path / "dense-cache"
            compact_cache = temp_path / "compact-cache"
            with dense_jsonl.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, record_with_categories("dense", (1, 2, 3, 4)))
            with compact_jsonl.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, record_with_categories("compact", (0, 5, 0, 6)))

            write_training_cache_from_rollouts(dense_jsonl, dense_cache, config=TrajectoryDatasetConfig(window_size=1))
            write_training_cache_from_rollouts(compact_jsonl, compact_cache, config=TrajectoryDatasetConfig(window_size=1))
            dense_metadata = json.loads((dense_cache / "metadata.json").read_text(encoding="utf-8"))
            compact_metadata = json.loads((compact_cache / "metadata.json").read_text(encoding="utf-8"))
            batch = next(
                iter_training_batches(
                    (dense_cache, compact_cache),
                    batch_size=2,
                    config=TrajectoryDatasetConfig(window_size=1),
                )
            )
            row_batch = next(
                iter_training_batches(
                    (dense_cache, compact_cache),
                    batch_size=2,
                    config=TrajectoryDatasetConfig(window_size=1),
                    defer_cache_window_expansion=True,
                )
            )

        self.assertEqual(dense_metadata["categorical_storage"]["stored_feature_count"], 4)
        self.assertEqual(compact_metadata["categorical_storage"]["stored_feature_count"], 2)
        self.assertEqual(batch.categorical_ids.shape[-1], 4)
        self.assertEqual(_tolist(batch.categorical_ids)[0][0][0], [1, 2, 3, 4])
        self.assertEqual(_tolist(batch.categorical_ids)[1][0][0], [5, 6, 0, 0])
        self.assertEqual(row_batch.categorical_ids, ())
        self.assertEqual(row_batch.row_categorical_ids.shape[-1], 4)
        self.assertEqual(_tolist(row_batch.row_categorical_ids)[0][0], [1, 2, 3, 4])
        self.assertEqual(_tolist(row_batch.row_categorical_ids)[1][0], [5, 6, 0, 0])
        self.assertEqual(_tolist(row_batch.window_row_indices), [[0], [1]])

    def test_training_cache_write_rejects_root_storage_cap_before_output_creation(self) -> None:
        self._require_numpy()
        builder = TrainingCacheBuilder(config=TrajectoryDatasetConfig(window_size=1))
        builder.add_record(rollout_record())
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"

            with self.assertRaisesRegex(ValueError, "storage cap"):
                builder.write(cache_path, max_cache_root_bytes=1, cache_root=temp_dir)

            self.assertFalse(cache_path.exists())

    def test_training_cache_write_root_cap_counts_existing_sibling_cache_bytes(self) -> None:
        self._require_numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            cache_root = Path(temp_dir) / "cache-root"
            first_cache = cache_root / "cache-000"
            second_cache = cache_root / "cache-001"
            cache_root.mkdir()
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            write_training_cache_from_rollouts(path, first_cache, config=TrajectoryDatasetConfig(window_size=1))
            existing_bytes = training_cache_root_byte_size(cache_root)

            with self.assertRaisesRegex(ValueError, "storage cap"):
                write_training_cache_from_rollouts(
                    path,
                    second_cache,
                    config=TrajectoryDatasetConfig(window_size=1),
                    max_cache_root_bytes=existing_bytes + 1,
                    cache_root=cache_root,
                )

            self.assertFalse(second_cache.exists())

    def test_training_cache_rejects_categorical_ids_outside_compact_range_before_cast(self) -> None:
        self._require_numpy()
        builder = TrainingCacheBuilder(config=TrajectoryDatasetConfig(window_size=1))
        builder.add_record(rollout_record())
        builder._categorical_rows[0] = _fill_like(builder._categorical_rows[0], 70_000)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "categorical ids"):
                builder.write(Path(temp_dir) / "cache")

    def test_training_cache_rejects_token_type_ids_outside_compact_range_before_cast(self) -> None:
        self._require_numpy()
        builder = TrainingCacheBuilder(config=TrajectoryDatasetConfig(window_size=1))
        builder.add_record(rollout_record())
        builder._token_type_rows[0] = tuple(300 for _ in builder._token_type_rows[0])
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "token type ids"):
                builder.write(Path(temp_dir) / "cache")

    def test_training_cache_rejects_mismatched_dataset_config(self) -> None:
        self._require_numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            cache_path = Path(temp_dir) / "cache"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            write_training_cache_from_rollouts(path, cache_path, config=TrajectoryDatasetConfig(window_size=2))

            with self.assertRaisesRegex(ValueError, "dataset config"):
                list(iter_training_batches(cache_path, batch_size=2, config=TrajectoryDatasetConfig(window_size=1)))

    def test_training_cache_rejects_mixed_cache_and_jsonl_paths(self) -> None:
        self._require_numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            cache_path = Path(temp_dir) / "cache"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            write_training_cache_from_rollouts(path, cache_path, config=TrajectoryDatasetConfig(window_size=1))

            with self.assertRaisesRegex(ValueError, "cannot be mixed"):
                list(iter_training_batches([path, cache_path], batch_size=2, config=TrajectoryDatasetConfig(window_size=1)))

    def test_training_cache_consumed_callback_fires_after_cache_read(self) -> None:
        self._require_numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            cache_path = Path(temp_dir) / "cache"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            write_training_cache_from_rollouts(path, cache_path, config=TrajectoryDatasetConfig(window_size=1))

            consumed: list[Path] = []
            batches = list(
                iter_training_batches(
                    cache_path,
                    batch_size=2,
                    config=TrajectoryDatasetConfig(window_size=1),
                    consumed_cache_callback=consumed.append,
                )
            )

            self.assertEqual([batch.batch_size for batch in batches], [2, 1])
            self.assertEqual(consumed, [cache_path])

    def test_training_cache_delete_helper_removes_only_cache_directory(self) -> None:
        self._require_numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            cache_path = Path(temp_dir) / "cache"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, rollout_record())
            write_training_cache_from_rollouts(path, cache_path, config=TrajectoryDatasetConfig(window_size=1))

            self.assertGreater(training_cache_paths_byte_size(cache_path), 0)
            delete_training_cache_path(cache_path)

            self.assertFalse(cache_path.exists())

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

    def _require_numpy(self) -> None:
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("NumPy is not installed in this environment.")


def _batch_payload(batches) -> list[dict]:
    return [
        {
            "categorical_ids": _tolist(batch.categorical_ids),
            "numeric_features": _tolist(batch.numeric_features),
            "token_type_ids": _tolist(batch.token_type_ids),
            "attention_mask": _tolist(batch.attention_mask),
            "history_mask": _tolist(batch.history_mask),
            "legal_action_mask": _tolist(batch.legal_action_mask),
            "action_indices": _tolist(batch.action_indices),
            "rewards": _tolist(batch.rewards),
            "returns": _tolist(batch.returns),
            "value_estimates": _tolist(batch.value_estimates),
            "value_estimate_mask": _tolist(batch.value_estimate_mask),
            "ppo_advantages": _tolist(batch.ppo_advantages),
            "ppo_advantage_mask": _tolist(batch.ppo_advantage_mask),
            "ppo_value_targets": _tolist(batch.ppo_value_targets),
            "ppo_value_target_mask": _tolist(batch.ppo_value_target_mask),
            "opponent_action_indices": _tolist(batch.opponent_action_indices),
            "opponent_action_mask": _tolist(batch.opponent_action_mask),
            "action_probabilities": _tolist(batch.action_probabilities),
            "action_probability_mask": _tolist(batch.action_probability_mask),
            "seeds": _tolist(batch.seeds),
            "turn_indices": _tolist(batch.turn_indices),
            "terminal_capped": _tolist(batch.terminal_capped),
        }
        for batch in batches
    ]


def _tolist(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_tolist(item) for item in value]
    if isinstance(value, list):
        return [_tolist(item) for item in value]
    return value


def _fill_like(value, replacement):
    if isinstance(value, tuple):
        return tuple(_fill_like(item, replacement) for item in value)
    return replacement


if __name__ == "__main__":
    unittest.main()
