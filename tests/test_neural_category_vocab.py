import json
from pathlib import Path
import tempfile
import unittest

from pokezero.neural_policy import (
    DEFAULT_CATEGORY_OOV_BUCKETS,
    TransformerPolicyConfig,
    collect_categorical_ids,
    torch_available,
)


def _write_rollout_jsonl(path: Path, categorical_rows) -> None:
    record = {
        "trajectory": {
            "steps": [
                {"observation": {"categorical_ids": categorical_rows}},
            ]
        }
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


class CollectCategoricalIdsTests(unittest.TestCase):
    def test_collects_distinct_nonzero_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout.jsonl"
            _write_rollout_jsonl(path, [[0, 5, 10], [10, 0, 42], [0, 0, 0]])
            self.assertEqual(collect_categorical_ids(path), (5, 10, 42))

    def test_accepts_multiple_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            _write_rollout_jsonl(a, [[0, 1]])
            _write_rollout_jsonl(b, [[2, 3]])
            self.assertEqual(collect_categorical_ids([a, b]), (1, 2, 3))


class CompactCategoryConfigTests(unittest.TestCase):
    def test_sizing_and_dedup_sort(self) -> None:
        config = TransformerPolicyConfig.compact_category(
            category_vocab=[20, 5, 10, 5, 0],
            category_oov_buckets=4,
        )
        self.assertEqual(config.category_vocab, (5, 10, 20))
        self.assertEqual(config.category_oov_buckets, 4)
        # 1 padding + 3 vocab + 4 oov
        self.assertEqual(config.categorical_vocab_size, 8)

    def test_serialization_round_trip(self) -> None:
        config = TransformerPolicyConfig.compact_category(category_vocab=[7, 3, 9], category_oov_buckets=8)
        restored = TransformerPolicyConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)

    def test_list_vocab_normalized_to_tuple_and_round_trips(self) -> None:
        config = TransformerPolicyConfig(categorical_vocab_size=4, category_vocab=[5, 10], category_oov_buckets=1)
        self.assertIsInstance(config.category_vocab, tuple)
        self.assertEqual(config.category_vocab, (5, 10))
        self.assertEqual(TransformerPolicyConfig.from_dict(config.to_dict()), config)

    def test_rejects_inconsistent_size(self) -> None:
        with self.assertRaises(ValueError):
            TransformerPolicyConfig(categorical_vocab_size=999, category_vocab=(5, 10), category_oov_buckets=4)

    def test_rejects_oov_without_vocab(self) -> None:
        with self.assertRaises(ValueError):
            TransformerPolicyConfig(category_oov_buckets=4)

    def test_default_oov_constant(self) -> None:
        config = TransformerPolicyConfig.compact_category(category_vocab=[1, 2])
        self.assertEqual(config.category_oov_buckets, DEFAULT_CATEGORY_OOV_BUCKETS)


@unittest.skipUnless(torch_available(), "requires torch")
class CompactRemapTests(unittest.TestCase):
    def _model(self, vocab, oov):
        import torch

        from pokezero.neural_policy import EntityTokenTransformerPolicy

        config = TransformerPolicyConfig.compact_category(category_vocab=vocab, category_oov_buckets=oov)
        return EntityTokenTransformerPolicy(config), torch

    def test_remap_is_lossless_and_deterministic(self) -> None:
        model, torch = self._model([5, 10, 20], 4)
        ids = torch.tensor([[0, 5, 10, 20, 7, 21]], dtype=torch.long)
        out = model._remap_category_ids(ids).tolist()[0]
        # padding stays 0; each in-vocab id gets a unique dedicated row (1..3)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 1)  # 5 -> slot 1
        self.assertEqual(out[2], 2)  # 10 -> slot 2
        self.assertEqual(out[3], 3)  # 20 -> slot 3
        # in-vocab rows are distinct (no collisions) => lossless
        self.assertEqual(len({out[1], out[2], out[3]}), 3)
        # out-of-vocab ids fold into the reserved oov block [4 .. 7]
        oov_base = 1 + 3
        self.assertEqual(out[4], oov_base + (7 % 4))
        self.assertEqual(out[5], oov_base + (21 % 4))
        for value in out:
            self.assertLess(value, model.config.categorical_vocab_size)

    def test_alias_remaps_to_base_slot(self) -> None:
        import torch

        from pokezero.neural_policy import EntityTokenTransformerPolicy

        # base ids 10 and 20 in vocab; 999 aliases onto base 10.
        config = TransformerPolicyConfig.compact_category(
            category_vocab=[10, 20], category_oov_buckets=4, category_aliases=[(999, 10)]
        )
        model = EntityTokenTransformerPolicy(config)
        ids = torch.tensor([[10, 20, 999]], dtype=torch.long)
        out = model._remap_category_ids(ids).tolist()[0]
        # 10 -> slot 1, 20 -> slot 2, and aliased 999 -> base 10's slot (1)
        self.assertEqual(out[0], 1)
        self.assertEqual(out[1], 2)
        self.assertEqual(out[2], out[0])

    def test_alias_validation_rejects_base_outside_vocab(self) -> None:
        with self.assertRaises(ValueError):
            TransformerPolicyConfig.compact_category(
                category_vocab=[10, 20], category_oov_buckets=4, category_aliases=[(999, 30)]
            )

    def test_embedding_is_compact(self) -> None:
        model, _ = self._model(list(range(1, 101)), 16)
        self.assertEqual(model.category_embedding.num_embeddings, 1 + 100 + 16)

    def test_buffer_not_persisted_but_rebuilt_from_config(self) -> None:
        model, _ = self._model([3, 8, 15], 4)
        # non-persistent buffer is excluded from the state dict (keeps checkpoints small)
        self.assertNotIn("category_vocab_sorted", model.state_dict())
        # but a model rebuilt from the serialized config remaps identically
        from pokezero.neural_policy import EntityTokenTransformerPolicy

        rebuilt = EntityTokenTransformerPolicy(TransformerPolicyConfig.from_dict(model.config.to_dict()))
        import torch

        ids = torch.tensor([[0, 3, 8, 15, 99]], dtype=torch.long)
        self.assertEqual(model._remap_category_ids(ids).tolist(), rebuilt._remap_category_ids(ids).tolist())


@unittest.skipUnless(torch_available(), "requires torch")
class CompactCheckpointRoundTripTests(unittest.TestCase):
    def test_save_load_preserves_vocab_and_remap(self) -> None:
        import torch

        from pokezero.neural_policy import (
            EntityTokenTransformerPolicy,
            TransformerEpochMetrics,
            TransformerTrainingConfig,
            TransformerTrainingResult,
            load_transformer_checkpoint,
            save_transformer_checkpoint,
        )

        config = TransformerPolicyConfig.compact_category(category_vocab=[3, 8, 15], category_oov_buckets=4)
        model = EntityTokenTransformerPolicy(config)
        result = TransformerTrainingResult(
            model_config=config,
            training_config=TransformerTrainingConfig(),
            epochs=(
                TransformerEpochMetrics(
                    epoch=1, examples=1, loss=1.0, policy_loss=1.0, policy_accuracy=0.5
                ),
            ),
        )
        ids = torch.tensor([[0, 3, 8, 15, 99, 4103]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compact.pt"
            save_transformer_checkpoint(path, model, result=result)
            # non-persistent remap buffer is not serialized, keeping the checkpoint small
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertNotIn("category_vocab_sorted", payload["state_dict"])
            # strict load succeeds and the rebuilt model remaps identically
            loaded, loaded_result = load_transformer_checkpoint(path)
            self.assertEqual(loaded_result.model_config.category_vocab, (3, 8, 15))
            self.assertEqual(
                loaded._remap_category_ids(ids).tolist(),
                model._remap_category_ids(ids).tolist(),
            )


if __name__ == "__main__":
    unittest.main()
