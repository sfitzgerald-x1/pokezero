"""batch_replay: bit-identity, single-stream proof, callback deferral, fail-closed.

The flag's contract: epochs 2+ replay the exact batch sequence epoch 1 streamed
(iteration is deterministic file-order — no shuffle exists in the dataset), so
training numerics are unchanged while the per-epoch re-read/re-collate cost is
removed. These tests pin that contract on CPU where determinism is exact.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero import neural_policy as neural_policy_module
from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.dataset import TrajectoryDatasetConfig, write_training_cache_from_rollouts
from pokezero.env import TerminalState
from pokezero.neural_policy import (
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    train_transformer_policy,
)
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep

try:  # torch-dependent suite
    import torch

    TORCH = True
except Exception:  # pragma: no cover
    TORCH = False

LEGAL_TWO_ACTION_MASK = (True, True, False, False, False, False, False, False, False)


def _observation(value: int) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((value,) for _ in range(spec.token_count)),
        numeric_features=tuple((float(value),) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=LEGAL_TWO_ACTION_MASK,
    )


def _rollout_record(seed: int) -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id=f"replay-{seed}", format_id="gen3randombattle", seed=seed)
    for turn_index in range(4):
        action_index = (turn_index + seed) % 2
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=turn_index,
                observation=_observation(action_index + 1),
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


def _build_caches(root: Path, count: int = 3) -> tuple[Path, ...]:
    config = TrajectoryDatasetConfig(window_size=2)
    caches = []
    for index in range(count):
        jsonl = root / f"rollouts-{index}.jsonl"
        with jsonl.open("w", encoding="utf-8") as handle:
            write_rollout_record(handle, _rollout_record(index + 1))
            write_rollout_record(handle, _rollout_record(index + 100))
        cache = root / f"cache-{index}"
        write_training_cache_from_rollouts(jsonl, cache, config=config)
        caches.append(cache)
    return tuple(caches)


def _model_config() -> TransformerPolicyConfig:
    # Dropout stays enabled on purpose: replay must not perturb the RNG stream,
    # so identically-seeded runs must produce identical dropout masks too.
    return TransformerPolicyConfig.compact_category(
        category_vocab=tuple(range(1, 17)), category_oov_buckets=4, policy_id="replay",
        window_size=2, token_type_vocab_size=8, categorical_feature_count=1,
        numeric_feature_count=1, embedding_dim=16, transformer_layers=1,
        attention_heads=4, feedforward_dim=32, dropout=0.1,
    )


def _training_config(**overrides) -> TransformerTrainingConfig:
    values = dict(batch_size=4, epochs=3, window_size=2, device="cpu")
    values.update(overrides)
    return TransformerTrainingConfig(**values)


@unittest.skipUnless(TORCH, "requires torch")
class BatchReplayTests(unittest.TestCase):
    def _train(self, caches, *, consumed=None, **config_overrides):
        torch.manual_seed(1234)
        kwargs = {}
        if consumed is not None:
            kwargs["consumed_cache_callback"] = consumed.append
        return train_transformer_policy(
            caches,
            model_config=_model_config(),
            training_config=_training_config(**config_overrides),
            **kwargs,
        )

    def test_replay_is_bit_identical_to_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            caches = _build_caches(Path(temp_dir))
            model_off, result_off = self._train(caches, batch_replay=False)
            model_on, result_on = self._train(caches, batch_replay=True)
        state_off = model_off.state_dict()
        state_on = model_on.state_dict()
        self.assertEqual(sorted(state_off), sorted(state_on))
        for key in state_off:
            self.assertTrue(torch.equal(state_off[key], state_on[key]), f"weight mismatch: {key}")
        for epoch_off, epoch_on in zip(result_off.epochs, result_on.epochs, strict=True):
            self.assertEqual(epoch_off.loss, epoch_on.loss)
            self.assertEqual(epoch_off.examples, epoch_on.examples)
            self.assertEqual(epoch_off.policy_loss, epoch_on.policy_loss)
            self.assertEqual(epoch_off.policy_accuracy, epoch_on.policy_accuracy)

    def test_replay_streams_the_caches_exactly_once(self) -> None:
        real_iter = neural_policy_module.iter_training_batches
        calls: list[int] = []

        def counting_iter(*args, **kwargs):
            calls.append(1)
            return real_iter(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            caches = _build_caches(Path(temp_dir))
            with patch.object(neural_policy_module, "iter_training_batches", counting_iter):
                self._train(caches, batch_replay=True)
                self.assertEqual(len(calls), 1)  # epoch 1 only
                calls.clear()
                self._train(caches, batch_replay=False)
                self.assertEqual(len(calls), 3)  # every epoch re-streams

    def test_consumed_cache_callback_same_paths_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            caches = _build_caches(Path(temp_dir))
            consumed_off: list[Path] = []
            consumed_on: list[Path] = []
            self._train(caches, consumed=consumed_off, batch_replay=False)
            self._train(caches, consumed=consumed_on, batch_replay=True)
        self.assertEqual(consumed_off, consumed_on)
        self.assertEqual(len(consumed_on), len(set(consumed_on)))  # once each

    def test_replay_fails_closed_with_auxiliary_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            caches = _build_caches(Path(temp_dir))
            torch.manual_seed(1234)
            with self.assertRaisesRegex(ValueError, "batch_replay"):
                train_transformer_policy(
                    caches[:2],
                    model_config=_model_config(),
                    training_config=_training_config(batch_replay=True),
                    auxiliary_paths=caches[2:],
                    auxiliary_max_fraction=0.1,
                )

    def test_single_epoch_replay_flag_is_inert(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            caches = _build_caches(Path(temp_dir))
            consumed: list[Path] = []
            _, result = self._train(caches, consumed=consumed, batch_replay=True, epochs=1)
        self.assertEqual(len(result.epochs), 1)
        self.assertEqual(len(consumed), len(caches))  # streamed callback fired as today


if __name__ == "__main__":
    unittest.main()
