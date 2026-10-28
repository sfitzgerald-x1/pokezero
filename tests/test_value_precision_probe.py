import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pokezero.value_precision_probe import (
    battle_split, calibration_metrics, frozen_token_features, hl_gauss_targets,
    make_value_head, outcome_labels, validate_cache_metadata,
)
from pokezero.neural_policy import torch_available


class OutcomeContractTests(unittest.TestCase):
    def test_discounted_outcomes_caps_and_ambiguous_zeros(self):
        # GAE targets can disagree in sign; this API takes only actual returns.
        labels, status = outcome_labels([0.9999 ** 500, -0.9999 ** 500, 0, -0.25],
                                         np.array([False, False, False, True]), {"discount": 0.9999})
        np.testing.assert_array_equal(status, [0, 0, 2, 1])
        np.testing.assert_array_equal(labels[:2], [1, -1])
        self.assertTrue(np.isnan(labels[2:]).all())

    def test_shaping_invalid_discount_and_corruption_refuse(self):
        for config in [{"discount": 0}, {"discount": 1.01}, {"discount": float("nan")},
                       {"discount": 1, "hp_delta_return_weight": 0.1},
                       {"discount": 1, "potential_shaping": {}}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                outcome_labels([1.0], np.array([False]), config)
        for values in [[float("nan")], [1.5]]:
            with self.assertRaises(ValueError):
                outcome_labels(values, np.array([False]), {"discount": 1})

    def test_battle_groups_remain_together(self):
        first = [battle_split(seed, lineage="same-lineage", salt="fixed") for seed in range(1000)]
        second = [battle_split(np.int64(seed), lineage="same-lineage", salt="fixed") for seed in range(1000)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "validation", "test"})
        with self.assertRaises(ValueError):
            battle_split(True, lineage="same-lineage", salt="fixed")

    def test_residual_identity_and_unmeasured_siblings(self):
        result = calibration_metrics([0.1, -0.2, 0.3], [1, -1, 0])
        self.assertAlmostEqual(result["mse"], result["bias_squared"] + result["centered_outcome_residual_variance"])
        self.assertEqual(result["top_two_action_resolution"]["status"], "UNMEASURED_REQUIRES_ACTUAL_SIBLING_OBSERVATIONS")

    def test_cohort_freeze_preserves_sources_battle_groups_and_unknowns(self):
        source = Path(__file__).resolve().parents[1] / "scripts/freeze_value_probe_cohort.py"
        spec = importlib.util.spec_from_file_location("freeze_cohort", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contract = {"belief_set_source_hash": "fixture", "feature_masks": {},
                    "categorical_vocab_size": 8, "token_type_vocab_size": 16}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            for iteration in range(3):
                path = root / f"iteration-{iteration:04}" / "shard-f0"
                path.mkdir(parents=True)
                arrays = {"returns": np.array([0.8, -0.5, 0, -0.25], np.float32),
                          "terminal_capped": np.array([False, False, False, True]),
                          "seeds": np.array([10, 11, 12, 13], np.int64),
                          "turn_indices": np.array([0, 0, 0, 0], np.int32),
                          "window_indices": np.arange(1, 5, dtype=np.uint32)[:, None],
                          "numeric_features": np.zeros((5, 23, 132), np.float16),
                          "categorical_ids": np.zeros((5, 23, 5), np.uint16),
                          "token_type_ids": np.zeros((5, 23), np.uint8),
                          "attention_mask": np.ones((5, 23), bool)}
                for key, value in arrays.items():
                    np.save(path / (key + ".npy"), value, allow_pickle=False)
                metadata = {"schema_version": "pokezero.training_cache.v2",
                            "observation_schema": "pokezero.observation.v4",
                            "collection_env": "showdown", "belief_fidelity": "full",
                            "belief_set_source_hash": "fixture", "feature_masks": {},
                            "example_count": 4, "dataset_config": {"discount": 0.99},
                            "categorical_storage": {"mode": "compact-nonzero", "original_feature_count": 41},
                            "observation_shapes": {"numeric_features": [23, 132], "categorical_ids": [23, 5],
                                "token_type_ids": [23], "attention_mask": [23], "window_size": 1}}
                (path / "metadata.json").write_text(json.dumps(metadata))
            original = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            first = module.freeze(root, Path(directory) / "first", contract, target=5,
                                  lineage="fixture", selection_salt="selection", split_salt="split")
            second = module.freeze(root, Path(directory) / "second", contract, target=5,
                                   lineage="fixture", selection_salt="selection", split_salt="split")
            self.assertEqual(first["selected_iterations"], second["selected_iterations"])
            self.assertEqual(first["selected_examples"], 8)  # complete iteration, no row trimming
            self.assertEqual(first["counts"], {"rows": 8, "known": 4, "capped": 2, "unresolved_zero": 2})
            for split, count in first["split_counts"].items():
                expected = sum(2 for seed in range(10, 14)
                               if battle_split(seed, lineage="fixture", salt="split") == split)
                self.assertEqual(count["rows"], expected)
            self.assertEqual(original, {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()})
            # A stale/malformed source is refused before membership is frozen.
            last = next(root.glob("iteration-*/shard-f0"))
            np.save(last / "seeds.npy", np.array([10, 11], np.int64))
            with self.assertRaises(ValueError):
                module.freeze(root, Path(directory) / "bad", contract, target=12,
                              lineage="fixture", selection_salt="selection", split_salt="split")
            np.save(last / "seeds.npy", np.array([10, 11, 12, 13], np.int64))
            # A valid-range row alias cannot move an observation across seed partitions.
            np.save(last / "window_indices.npy", np.array([[1], [1], [3], [4]], np.uint32))
            with self.assertRaises(ValueError):
                module.freeze(root, Path(directory) / "aliased", contract, target=12,
                              lineage="fixture", selection_salt="selection", split_salt="split")
            np.save(last / "window_indices.npy", np.arange(1, 5, dtype=np.uint32)[:, None])
            # One incompatible shard excludes its ENTIRE iteration before label inspection.
            incompatible = last.parent / "shard-incompatible"
            incompatible.mkdir()
            metadata["feature_masks"] = {"incompatible": True}
            (incompatible / "metadata.json").write_text(json.dumps(metadata))
            mixed = module.freeze(root, Path(directory) / "mixed", contract, target=5,
                                  lineage="fixture", selection_salt="selection", split_salt="split")
            self.assertNotIn(last.parent.name, mixed["selected_iterations"])


@unittest.skipUnless(torch_available(), "PyTorch required")
class ValueHeadTests(unittest.TestCase):
    def test_hl_gauss_boundary_mass_symmetry_and_gradients(self):
        from pokezero.neural_policy import require_torch
        torch = require_torch()
        labels = hl_gauss_targets(torch.tensor([-1.0, 0.0, 1.0]))
        torch.testing.assert_close(labels.sum(-1), torch.ones(3))
        torch.testing.assert_close(labels[0], labels[2].flip(0))
        torch.testing.assert_close(labels[1], labels[1].flip(0))
        logits = torch.zeros_like(labels, requires_grad=True)
        loss = -(labels * logits.log_softmax(-1)).sum(-1).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            hl_gauss_targets(torch.tensor([1.1]))

    def test_all_heads_ignore_padded_tokens_and_can_learn(self):
        from pokezero.neural_policy import require_torch
        torch = require_torch()
        torch.manual_seed(9)
        tokens = torch.randn(3, 5, 8)
        valid = torch.tensor([[True, True, False, False, False]] * 3)
        altered = tokens.clone()
        altered[:, 2:] = 100000
        for kind in ["linear", "attention_mlp", "attention_hl_gauss"]:
            with self.subTest(kind=kind):
                head = make_value_head(kind, 8, hidden=12, tanh_value=True)
                before = {k: v.clone() for k, v in head.state_dict().items()}
                value, logits = head(tokens, valid)
                torch.testing.assert_close(value, head(altered, valid)[0])
                target = torch.tensor([1.0, -1.0, 1.0])
                loss = ((value - target) ** 2).mean() if logits is None else -(hl_gauss_targets(target) * logits.log_softmax(-1)).sum(-1).mean()
                optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.assertTrue(any(not torch.equal(v, before[k]) for k, v in head.state_dict().items()))

    def test_frozen_capture_reproduces_production_mean_head_and_trunk(self):
        from pokezero.neural_policy import EntityTokenTransformerPolicy, TransformerPolicyConfig, require_torch
        torch = require_torch()
        torch.manual_seed(11)
        config = TransformerPolicyConfig.compact_category(category_vocab=["kind:a", "kind:b"],
            window_size=1, embedding_dim=16, attention_heads=4,
            feedforward_dim=32, transformer_layers=1, dropout=0.0,
            observation_schema_version="pokezero.observation.v4")
        model = EntityTokenTransformerPolicy(config)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        tensors = {"categorical_ids": torch.randint(0, config.categorical_vocab_size, (2, 1, 23, 41)),
                   "numeric_features": torch.randn(2, 1, 23, 132),
                   "token_type_ids": torch.zeros(2, 1, 23, dtype=torch.long),
                   "attention_mask": torch.ones(2, 1, 23, dtype=torch.bool),
                   "history_mask": torch.ones(2, 1, dtype=torch.bool)}
        tensors["attention_mask"][:, :, 10:13] = False
        tokens, output = frozen_token_features(model, tensors)
        valid = tensors["attention_mask"][:, 0]
        pooled = torch.where(valid[..., None], tokens, 0).sum(1) / valid.sum(1, keepdim=True)
        expected = torch.tanh(model.value_head(pooled).squeeze(-1))
        torch.testing.assert_close(output.value, expected)
        head = make_value_head("attention_mlp", 16, hidden=16)
        head(tokens, valid)[0].sum().backward()
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
