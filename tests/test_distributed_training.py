"""Parity coverage for the opt-in one-node DDP trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import tempfile
import unittest

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero import neural_cli
from pokezero.env import TerminalState
from pokezero.neural_policy import (
    EntityTokenTransformerPolicy,
    DistributedTrainingContext,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    initialize_distributed_training,
    load_transformer_checkpoint,
    save_transformer_checkpoint,
    torch_available,
    train_transformer_policy,
    _distributed_loss_term,
)
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _observation(value: int) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((value,) for _ in range(spec.token_count)),
        numeric_features=tuple((float(value),) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=(True, True, False, False, False, False, False, False, False),
    )


def _rollout_record() -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id="ddp-parity", format_id="gen3randombattle", seed=17)
    # Nine examples with global batches of four forces a one-example final
    # batch: rank 0 owns it while rank 1 runs only the zero-weight DDP
    # placeholder. This is the ragged-final-batch invariant from WS-A2.
    for turn_index in range(9):
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=turn_index,
                observation=_observation((turn_index % 3) + 1),
                legal_action_mask=(True, True, False, False, False, False, False, False, False),
                action_index=turn_index % 2,
                opponent_action_index=1 - (turn_index % 2),
                action_probability=0.5,
                value_estimate=0.0,
            )
        )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=9))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "fixture"},
        decision_round_count=9,
        elapsed_seconds=0.01,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def _ranking_rollout_record() -> RolloutRecord:
    """Five returns that make two rank-local pairs plus a ragged final shard."""

    trajectory = BattleTrajectory(battle_id="ddp-ranking", format_id="gen3randombattle", seed=19)
    # With a global batch size of four and two ranks, the first batch gives
    # each rank one p1 (+1) and one p2 (-1) decision: exactly one local pair
    # per rank. The fifth decision exercises the zero-example placeholder on
    # rank 1 without inventing a pair.
    for turn_index, player_id in enumerate(("p1", "p2", "p1", "p2", "p1")):
        trajectory.append(
            TrajectoryStep(
                player_id=player_id,
                turn_index=turn_index,
                observation=_observation((turn_index % 3) + 1),
                legal_action_mask=(True, True, False, False, False, False, False, False, False),
                action_index=turn_index % 2,
                opponent_action_index=1 - (turn_index % 2),
                action_probability=0.5,
                value_estimate=0.0,
            )
        )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=5))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "fixture", "p2": "fixture"},
        decision_round_count=5,
        elapsed_seconds=0.01,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def _ddp_train_worker(
    rank: int,
    world_size: int,
    port: int,
    data_path: str,
    initial_checkpoint: str,
    output_checkpoint: str,
    training_options: dict[str, object],
) -> None:
    # Spawned processes inherit pytest's environment; torchrun's three rank
    # variables are the only input the production trainer relies on.
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    context = initialize_distributed_training("cpu")
    try:
        model, prior = load_transformer_checkpoint(initial_checkpoint, map_location="cpu")
        trained, result = train_transformer_policy(
            data_path,
            model_config=prior.model_config,
            initial_model=model,
            training_config=TransformerTrainingConfig(**training_options),
            distributed_context_override=context,
        )
        if context.is_primary:
            save_transformer_checkpoint(output_checkpoint, trained, result=result)
        import torch

        torch.distributed.barrier()
    finally:
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _ddp_value_selection_worker(
    rank: int,
    world_size: int,
    port: int,
    data_path: str,
    initial_checkpoint: str,
    selection_summary: str,
) -> None:
    """Exercise the rank-zero-only callback path used by Phase-3 torchrun."""

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    context = initialize_distributed_training("cpu")
    try:
        model, prior = load_transformer_checkpoint(initial_checkpoint, map_location="cpu")
        selected_model, selected_result, selection = neural_cli._train_with_value_selection(
            paths=[Path(data_path)],
            model_config=prior.model_config,
            training_config=TransformerTrainingConfig(
                batch_size=4,
                epochs=1,
                max_batches=1,
                window_size=1,
                device="cpu",
                value_loss_weight=0.25,
                opponent_action_loss_weight=0.0,
                objective="behavior-cloning",
                random_seed=113,
            ),
            initial_model=model,
            selection_paths=[Path(data_path)],
            selection_metric="mse",
            batch_size=4,
            bins=10,
            distributed_context_override=context,
        )
        if context.is_primary:
            if not selection:
                raise AssertionError("rank zero must publish a selection witness")
            save_transformer_checkpoint(selection_summary + ".pt", selected_model, result=selected_result)
            Path(selection_summary).write_text(
                json.dumps(
                    {
                        "selected_epoch": selection["selected_epoch"],
                        "epochs": len(selection["epochs"]),
                        "examples": selected_result.final_metrics.examples,
                    }
                ),
                encoding="utf-8",
            )
        elif selection:
            raise AssertionError("non-primary rank must not manufacture a selection witness")
        import torch

        torch.distributed.barrier()
    finally:
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@unittest.skipUnless(torch_available(), "requires torch")
class DistributedTrainingTest(unittest.TestCase):
    def test_distributed_loss_term_preserves_legacy_denominator_floor(self) -> None:
        """A sparse fractional weighted batch must retain single-process scaling."""

        import torch

        numerator = torch.tensor(0.25, requires_grad=True)
        gradient_term, metric, raw_denominator = _distributed_loss_term(
            numerator,
            torch.tensor(0.5),
            context=DistributedTrainingContext(),
        )
        gradient_term.backward()
        self.assertEqual(raw_denominator, 0.5)
        self.assertEqual(metric, 0.25)
        self.assertEqual(float(numerator.grad.item()), 1.0)

    def test_ddp_two_rank_contiguous_shards_match_single_device_updates(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "rollouts.jsonl"
            initial_path = root / "initial.pt"
            single_path = root / "single.pt"
            ddp_path = root / "ddp.pt"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, _rollout_record())
            config = TransformerPolicyConfig.compact_category(
                category_vocab=("a", "b", "c", "d"),
                category_oov_buckets=1,
                policy_id="ddp-parity",
                window_size=1,
                categorical_feature_count=1,
                numeric_feature_count=1,
                embedding_dim=8,
                # The strict parameter-delta gate deliberately avoids attention
                # kernel reduction-order drift; the PPO parity gate below runs
                # a real transformer layer.
                transformer_layers=0,
                attention_heads=1,
                feedforward_dim=8,
                dropout=0.0,
            )
            torch.manual_seed(91)
            initial_model = EntityTokenTransformerPolicy(config)
            empty_result = TransformerTrainingResult(
                model_config=config,
                training_config=TransformerTrainingConfig(batch_size=4, window_size=1, device="cpu"),
                epochs=(),
            )
            save_transformer_checkpoint(initial_path, initial_model, result=empty_result)

            single_initial, _ = load_transformer_checkpoint(initial_path, map_location="cpu")
            training_options = {
                "batch_size": 4,
                "epochs": 3,
                "window_size": 1,
                "device": "cpu",
                "value_loss_weight": 0.25,
                "opponent_action_loss_weight": 0.0,
                "objective": "behavior-cloning",
                "random_seed": 91,
            }
            single_model, single_result = train_transformer_policy(
                data_path,
                model_config=config,
                initial_model=single_initial,
                training_config=TransformerTrainingConfig(**training_options),
            )
            save_transformer_checkpoint(single_path, single_model, result=single_result)

            torch.multiprocessing.spawn(
                _ddp_train_worker,
                args=(2, _free_local_port(), str(data_path), str(initial_path), str(ddp_path), training_options),
                nprocs=2,
                join=True,
            )
            ddp_model, ddp_result = load_transformer_checkpoint(ddp_path, map_location="cpu")
            self.assertEqual(ddp_result.final_metrics.examples, single_result.final_metrics.examples)
            self.assertAlmostEqual(ddp_result.final_metrics.loss, single_result.final_metrics.loss, places=6)
            for name, value in single_model.state_dict().items():
                self.assertTrue(
                    torch.allclose(value, ddp_model.state_dict()[name], rtol=0.0, atol=1e-6),
                    name,
                )

    def test_ddp_value_selection_returns_only_rank_zero_witness(self) -> None:
        """All ranks participate in training while only rank zero selects an epoch."""
        import torch

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "rollouts.jsonl"
            initial_path = root / "initial.pt"
            selection_summary = root / "selection.json"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, _rollout_record())
            config = TransformerPolicyConfig.compact_category(
                category_vocab=("a", "b", "c", "d"),
                category_oov_buckets=1,
                policy_id="ddp-value-selection",
                window_size=1,
                categorical_feature_count=1,
                numeric_feature_count=1,
                embedding_dim=8,
                transformer_layers=0,
                attention_heads=1,
                feedforward_dim=8,
                dropout=0.0,
            )
            torch.manual_seed(113)
            initial_model = EntityTokenTransformerPolicy(config)
            save_transformer_checkpoint(
                initial_path,
                initial_model,
                result=TransformerTrainingResult(
                    model_config=config,
                    training_config=TransformerTrainingConfig(batch_size=4, window_size=1, device="cpu"),
                    epochs=(),
                ),
            )
            torch.multiprocessing.spawn(
                _ddp_value_selection_worker,
                args=(2, _free_local_port(), str(data_path), str(initial_path), str(selection_summary)),
                nprocs=2,
                join=True,
            )
            self.assertEqual(
                json.loads(selection_summary.read_text(encoding="utf-8")),
                {"selected_epoch": 1, "epochs": 1, "examples": 4},
            )

    def test_ddp_value_ranking_uses_only_rank_local_pairs(self) -> None:
        """V2 ranking permits DDP without silently creating cross-rank pairs."""

        import torch

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "ranking-rollouts.jsonl"
            initial_path = root / "initial.pt"
            ddp_path = root / "ddp.pt"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, _ranking_rollout_record())
            config = TransformerPolicyConfig.compact_category(
                category_vocab=("a", "b", "c", "d"),
                category_oov_buckets=1,
                policy_id="ddp-rank-local-ranking",
                window_size=1,
                categorical_feature_count=1,
                numeric_feature_count=1,
                embedding_dim=8,
                transformer_layers=0,
                attention_heads=1,
                feedforward_dim=8,
                dropout=0.0,
            )
            torch.manual_seed(101)
            initial_model = EntityTokenTransformerPolicy(config)
            save_transformer_checkpoint(
                initial_path,
                initial_model,
                result=TransformerTrainingResult(
                    model_config=config,
                    training_config=TransformerTrainingConfig(batch_size=4, window_size=1, device="cpu"),
                    epochs=(),
                ),
            )
            training_options = {
                "batch_size": 4,
                "epochs": 1,
                "window_size": 1,
                "device": "cpu",
                "value_loss_weight": 0.25,
                "value_ranking_loss_weight": 0.5,
                "value_ranking_margin": 0.1,
                "opponent_action_loss_weight": 0.0,
                "objective": "behavior-cloning",
                "random_seed": 101,
            }
            torch.multiprocessing.spawn(
                _ddp_train_worker,
                args=(2, _free_local_port(), str(data_path), str(initial_path), str(ddp_path), training_options),
                nprocs=2,
                join=True,
            )
            _, result = load_transformer_checkpoint(ddp_path, map_location="cpu")
            metrics = result.final_metrics
            self.assertEqual(metrics.examples, 5)
            # Four global examples would make four opposing-return pairs. V2
            # instead makes one within each rank's two-example shard, and the
            # ragged final global batch has no valid pair.
            self.assertEqual(metrics.value_ranking_pairs, 2)
            self.assertIsNotNone(metrics.value_ranking_loss)
            self.assertGreater(float(metrics.value_ranking_loss), 0.0)

    def test_ddp_two_rank_ppo_reports_global_metrics_within_parity_bounds(self) -> None:
        """The PPO path allows reduction-order drift but not metric drift.

        AdamW can amplify sub-ULP all-reduce differences on near-zero gradients, so
        the recipe's practical gate is global PPO metric parity rather than
        bitwise-identical parameters. The strict parameter gate above still
        protects the deterministic non-PPO reduction mechanism.
        """

        import torch

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "rollouts.jsonl"
            initial_path = root / "initial.pt"
            ddp_path = root / "ddp.pt"
            with data_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, _rollout_record())
            config = TransformerPolicyConfig.compact_category(
                category_vocab=("a", "b", "c", "d"),
                category_oov_buckets=1,
                policy_id="ddp-ppo-parity",
                window_size=1,
                categorical_feature_count=1,
                numeric_feature_count=1,
                embedding_dim=8,
                transformer_layers=1,
                attention_heads=2,
                feedforward_dim=16,
                dropout=0.0,
            )
            torch.manual_seed(91)
            initial_model = EntityTokenTransformerPolicy(config)
            save_transformer_checkpoint(
                initial_path,
                initial_model,
                result=TransformerTrainingResult(
                    model_config=config,
                    training_config=TransformerTrainingConfig(batch_size=4, window_size=1, device="cpu"),
                    epochs=(),
                ),
            )
            training_options = {
                "batch_size": 4,
                "epochs": 1,
                "max_batches": 1,
                "window_size": 1,
                "device": "cpu",
                "value_loss_weight": 0.25,
                "value_clip_range": 0.0184,
                "opponent_action_loss_weight": 0.1,
                "objective": "ppo",
                "entropy_coef": 0.01,
                "random_seed": 91,
            }
            single_initial, _ = load_transformer_checkpoint(initial_path, map_location="cpu")
            _, single_result = train_transformer_policy(
                data_path,
                model_config=config,
                initial_model=single_initial,
                training_config=TransformerTrainingConfig(**training_options),
            )
            torch.multiprocessing.spawn(
                _ddp_train_worker,
                args=(2, _free_local_port(), str(data_path), str(initial_path), str(ddp_path), training_options),
                nprocs=2,
                join=True,
            )
            _, ddp_result = load_transformer_checkpoint(ddp_path, map_location="cpu")
            single = single_result.final_metrics
            ddp = ddp_result.final_metrics
            self.assertEqual(ddp.examples, single.examples)
            self.assertAlmostEqual(ddp.ppo_ratio_mean, single.ppo_ratio_mean, delta=0.005)
            self.assertLessEqual(abs(float(ddp.ppo_clip_fraction) - float(single.ppo_clip_fraction)), 0.02)
            self.assertLessEqual(abs(float(ddp.ppo_entropy) - float(single.ppo_entropy)), 0.02)
            self.assertAlmostEqual(ddp.value_loss, single.value_loss, delta=0.01)
