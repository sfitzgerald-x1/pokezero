"""Independent hand computation and real distributed reductions for search targets."""
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import socket
import tempfile
import unittest

import torch

from pokezero.neural_policy import (
    DistributedTrainingContext, TransformerPolicyOutput, TransformerTrainingConfig,
    _distributed_transformer_loss, _training_value_targets, _transformer_loss,
    _TorchMetricTotals,
)


def fixture():
    t = {
        "legal_action_mask": torch.tensor([[True, True] + [False] * 7] * 3),
        "action_indices": torch.tensor([0, 1, 0]),
        "returns": torch.tensor([1.0, -1.0, 0.5]),
        "value_estimates": torch.tensor([0.1, -0.2, 0.3]),
        "value_estimate_mask": torch.ones(3, dtype=torch.bool),
        "ppo_advantages": torch.tensor([0.5, -0.3, 0.1]),
        "ppo_advantage_mask": torch.ones(3, dtype=torch.bool),
        "ppo_value_targets": torch.tensor([0.8, -0.6, 0.2]),
        "ppo_value_target_mask": torch.tensor([True, True, False]),
        "action_probabilities": torch.tensor([0.6, 0.4, 0.5]),
        "action_probability_mask": torch.ones(3, dtype=torch.bool),
        "opponent_action_mask": torch.zeros(3, dtype=torch.bool),
        "opponent_action_indices": torch.zeros(3, dtype=torch.long),
        "training_weights": torch.tensor([1.0, 2.0, 0.5]),
        # A deliberately poisoned UNLABELED row must never enter arithmetic.
        "search_policy_targets": torch.tensor([[0.25, 0.75] + [0.] * 7,
                                                [float("nan")] * 9, [0.6, 0.4] + [0.] * 7]),
        "search_policy_target_mask": torch.tensor([True, False, True]),
        "search_value_targets": torch.tensor([-0.4, float("nan"), 0.7]),
        "search_value_target_mask": torch.tensor([True, False, True]),
    }
    output = TransformerPolicyOutput(
        policy_logits=torch.tensor([[0.2, -0.1] + [2.] * 7, [0.4, 0.6] + [3.] * 7,
                                    [-0.2, 0.3] + [1.] * 7], requires_grad=True),
        value=torch.tensor([0.3, -0.4, 0.1], requires_grad=True),
        opponent_action_logits=torch.zeros((3, 9), requires_grad=True),
    )
    cfg = TransformerTrainingConfig(objective="ppo", normalize_advantage=False, entropy_coef=0,
                                    value_loss_weight=0.4375, opponent_action_loss_weight=0,
                                    search_policy_loss_weight=1.0, search_value_mix_weight=0.5)
    return output, t, cfg


def reference(output, t, *, clipped=False):
    # Deliberately compute from literal target/advantage values, without production helpers.
    lp = output.policy_logits[:, :2].log_softmax(1)
    chosen = lp[torch.arange(3), torch.tensor([0, 1, 0])]
    ratio = chosen.exp() / torch.tensor([0.6, 0.4, 0.5])
    advantage = torch.tensor([0.5, -0.3, 0.1]); weight = torch.tensor([1., 2., 0.5])
    ppo = -(torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage) * weight).sum() / 3.5
    targets = torch.tensor([0.2, -0.6, 0.6])
    residual = (output.value - targets).square()
    if clipped:
        old = torch.tensor([0.1, -0.2, 0.3]); prediction = old + (output.value - old).clamp(-0.1, 0.1)
        residual = torch.maximum(residual, (prediction - targets).square())
    value = (residual * weight).sum() / 3.5
    search = -(lp[0] * torch.tensor([0.25, 0.75])).sum()
    search += -0.5 * (lp[2] * torch.tensor([0.6, 0.4])).sum()
    search /= 1.5
    return ppo + 0.4375 * value + search


def distributed_worker(rank, port, root):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                                         rank=rank, world_size=2, timeout=timedelta(seconds=25))
    try:
        # Run uneven label counts, a rank with zero labels, then a wholly inactive rank.
        for case in ("uneven", "zero_labels", "inactive"):
            out, t, cfg = fixture()
            if case == "uneven":
                sl = slice(0, 2) if rank == 0 else slice(2, 3); active = 2 if rank == 0 else 1
            elif case == "zero_labels":
                sl = torch.tensor([0, 2]) if rank == 0 else torch.tensor([1]); active = 2 if rank == 0 else 1
            else:
                sl = slice(0, 3) if rank == 0 else slice(1, 2); active = 3 if rank == 0 else (1 if case == "zero_labels" else 0)
                if rank == 1:
                    t["training_weights"][1] = 0.0
            tensors = {k: v[sl] for k, v in t.items()}
            logits = out.policy_logits[sl].detach().requires_grad_(True)
            value = out.value[sl].detach().requires_grad_(True)
            local = TransformerPolicyOutput(logits, value, out.opponent_action_logits[sl])
            loss, pieces = _distributed_transformer_loss(local, tensors, cfg,
                context=DistributedTrainingContext(rank=rank, world_size=2, backend="gloo"),
                local_examples=active, global_examples=3)
            loss.backward()
            # Mimic DDP's parameter-gradient average for each disjoint row.
            pg = torch.zeros_like(out.policy_logits); vg = torch.zeros_like(out.value)
            pg[sl] = logits.grad / 2; vg[sl] = value.grad / 2
            torch.distributed.all_reduce(pg); torch.distributed.all_reduce(vg)
            if rank == 0:
                Path(root, case + ".json").write_text(json.dumps({"policy_grad": pg.tolist(),
                    "value_grad": vg.tolist(), "loss": pieces["loss"]}))
    finally:
        torch.distributed.destroy_process_group()


class SearchLabelLossTest(unittest.TestCase):
    def test_hand_computed_loss_and_gradients_with_gae_and_independent_masks(self):
        for clipped in (False, True):
            with self.subTest(clipped=clipped):
                out, t, cfg = fixture()
                if clipped:
                    cfg = replace(cfg, value_clip_range=0.1)
                snapshots = {k: v.clone() for k, v in t.items() if not k.startswith("search_")}
                actual, pieces = _transformer_loss(out, t, cfg)
                expected = reference(out, t, clipped=clipped)
                torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
                gradients = torch.autograd.grad(actual, (out.policy_logits, out.value), retain_graph=True)
                reference_gradients = torch.autograd.grad(expected, (out.policy_logits, out.value))
                for a, b in zip(gradients, reference_gradients):
                    torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
                self.assertEqual(pieces["search_policy_weight_sum"], 1.5)
                for k, v in snapshots.items():
                    self.assertTrue(torch.equal(t[k], v), k)

    def test_zero_weights_ignore_labels_with_bit_identical_loss_gradients_and_metrics(self):
        out, t, cfg = fixture(); cfg = replace(cfg, search_policy_loss_weight=0, search_value_mix_weight=0)
        old = {k: v for k, v in t.items() if not k.startswith("search_")}
        a, ma = _transformer_loss(out, old, cfg); b, mb = _transformer_loss(out, t, cfg)
        self.assertTrue(torch.equal(a, b)); self.assertEqual(ma, mb)
        for x, y in zip(torch.autograd.grad(a, (out.policy_logits, out.value), retain_graph=True),
                        torch.autograd.grad(b, (out.policy_logits, out.value))):
            self.assertTrue(torch.equal(x, y))
        self.assertNotIn("search_policy_loss_weight", cfg.to_dict())
        self.assertNotIn("search_value_mix_weight", cfg.to_dict())

    def test_value_mix_keeps_policy_advantages_and_uses_masks(self):
        out, t, cfg = fixture()
        torch.testing.assert_close(_training_value_targets(t, cfg), torch.tensor([0.2, -0.6, 0.6]))
        _, before = _transformer_loss(out, t, replace(cfg, search_policy_loss_weight=0, search_value_mix_weight=0))
        _, after = _transformer_loss(out, t, replace(cfg, search_policy_loss_weight=0))
        self.assertEqual(before["policy_loss"], after["policy_loss"])
        self.assertEqual(before["ppo_advantage_sum"], after["ppo_advantage_sum"])

    def test_absent_or_zero_labels_remain_finite(self):
        for absent in (False, True):
            out, t, cfg = fixture()
            if absent:
                t = {k: v for k, v in t.items() if not k.startswith("search_")}
            else:
                t["search_policy_target_mask"].fill_(False); t["search_value_target_mask"].fill_(False)
            loss, pieces = _transformer_loss(out, t, cfg)
            self.assertTrue(torch.isfinite(loss)); self.assertEqual(pieces["search_policy_loss"], 0)

    def test_config_refuses_invalid_weights_and_non_ppo_use(self):
        for options in ({"search_policy_loss_weight": -1}, {"search_policy_loss_weight": float("nan")},
                        {"search_value_mix_weight": 1.1}, {"search_value_mix_weight": float("inf")},
                        {"search_value_mix_weight": -0.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TransformerTrainingConfig(objective="ppo", **options)
        with self.assertRaises(ValueError):
            TransformerTrainingConfig(search_policy_loss_weight=1)

    def test_epoch_summaries_retain_weighted_numerator_counts_and_off_identity(self):
        out, t, cfg = fixture(); _, pieces = _transformer_loss(out, t, cfg)
        totals = _TorchMetricTotals(); totals.add(3, pieces); totals.add(3, pieces)
        metrics = totals.to_epoch_metrics(1).to_dict()
        self.assertEqual(metrics["search_policy_weight_sum"], 3.0)
        self.assertEqual(metrics["search_policy_examples"], 4)
        self.assertEqual(metrics["search_value_examples"], 4)
        self.assertAlmostEqual(metrics["search_policy_loss"], pieces["search_policy_loss"], places=7)
        _, off = _transformer_loss(out, t, replace(cfg, search_policy_loss_weight=0, search_value_mix_weight=0))
        totals = _TorchMetricTotals(); totals.add(3, off)
        self.assertFalse(any(k.startswith("search_") for k in totals.to_epoch_metrics(1).to_dict()))

    def test_label_config_and_epoch_metrics_round_trip_through_unit_checkpoint(self):
        from pokezero.neural_policy import (
            EntityTokenTransformerPolicy, TransformerPolicyConfig, TransformerTrainingResult,
            load_transformer_checkpoint, save_transformer_checkpoint,
        )
        out, tensors, cfg = fixture(); _, pieces = _transformer_loss(out, tensors, cfg)
        totals = _TorchMetricTotals(); totals.add(3, pieces)
        model_cfg = TransformerPolicyConfig.compact_category(
            category_vocab=("unit",), category_oov_buckets=1, embedding_dim=8,
            transformer_layers=0, attention_heads=1, feedforward_dim=8,
            categorical_feature_count=1, numeric_feature_count=1, dropout=0)
        model = EntityTokenTransformerPolicy(model_cfg)
        result = TransformerTrainingResult(model_config=model_cfg, training_config=cfg,
                                            epochs=(totals.to_epoch_metrics(1),))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "unit.pt"; save_transformer_checkpoint(path, model, result=result)
            _, loaded = load_transformer_checkpoint(path, map_location="cpu")
            self.assertEqual(loaded.training_config.to_dict(), cfg.to_dict())
            self.assertEqual(loaded.final_metrics.to_dict(), result.final_metrics.to_dict())

    def test_real_two_rank_reduction_matches_single_device_with_empty_and_unlabeled_rank(self):
        with tempfile.TemporaryDirectory() as root:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0)); port = listener.getsockname()[1]
            torch.multiprocessing.spawn(distributed_worker, args=(port, root), nprocs=2, join=True)
            out, t, cfg = fixture(); expected = reference(out, t)
            pg, vg = torch.autograd.grad(expected, (out.policy_logits, out.value))
            for case in ("uneven", "zero_labels", "inactive"):
                actual = json.loads(Path(root, case + ".json").read_text())
                torch.testing.assert_close(torch.tensor(actual["policy_grad"]), pg, rtol=0, atol=1e-7)
                torch.testing.assert_close(torch.tensor(actual["value_grad"]), vg, rtol=0, atol=1e-7)
                self.assertAlmostEqual(actual["loss"], expected.item(), places=6)
