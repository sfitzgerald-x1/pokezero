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
    """The category vocabulary is now a direct string->row map (no 1M-hash, no in-model remap)."""

    def test_sizing_and_dedup_sort(self) -> None:
        config = TransformerPolicyConfig.compact_category(
            category_vocab=["type:fire", "move:psychic", "TYPE:fire", "species:charizard"],
            category_oov_buckets=4,
        )
        # normalized + deduped + sorted
        self.assertEqual(config.category_vocab, ("move:psychic", "species:charizard", "type:fire"))
        self.assertEqual(config.category_oov_buckets, 4)
        # 1 padding + 3 vocab + 4 oov
        self.assertEqual(config.categorical_vocab_size, 8)

    def test_serialization_round_trip(self) -> None:
        config = TransformerPolicyConfig.compact_category(
            category_vocab=["move:psychic", "species:blissey", "type:water"], category_oov_buckets=8
        )
        restored = TransformerPolicyConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)

    def test_list_vocab_normalized_to_tuple_and_round_trips(self) -> None:
        config = TransformerPolicyConfig(
            categorical_vocab_size=4, category_vocab=["species:a", "species:b"], category_oov_buckets=1
        )
        self.assertIsInstance(config.category_vocab, tuple)
        self.assertEqual(config.category_vocab, ("species:a", "species:b"))
        self.assertEqual(TransformerPolicyConfig.from_dict(config.to_dict()), config)

    def test_rejects_inconsistent_size(self) -> None:
        with self.assertRaises(ValueError):
            TransformerPolicyConfig(
                categorical_vocab_size=999, category_vocab=("species:a", "species:b"), category_oov_buckets=4
            )

    def test_rejects_oov_without_vocab(self) -> None:
        with self.assertRaises(ValueError):
            TransformerPolicyConfig(category_oov_buckets=4)

    def test_default_oov_constant(self) -> None:
        config = TransformerPolicyConfig.compact_category(category_vocab=["species:a", "species:b"])
        self.assertEqual(config.category_oov_buckets, DEFAULT_CATEGORY_OOV_BUCKETS)


@unittest.skipUnless(torch_available(), "requires torch")
class CompactEmbeddingTests(unittest.TestCase):
    """The model embeds the pre-converted rows directly (no remap)."""

    def _model(self, vocab, oov):
        from pokezero.neural_policy import EntityTokenTransformerPolicy

        config = TransformerPolicyConfig.compact_category(category_vocab=vocab, category_oov_buckets=oov)
        return EntityTokenTransformerPolicy(config)

    def test_embedding_size_matches_vocab(self) -> None:
        model = self._model([f"species:s{i}" for i in range(100)], 16)
        self.assertEqual(model.category_embedding.num_embeddings, 1 + 100 + 16)

    def test_no_remap_method(self) -> None:
        # The in-model hash->row remap is retired; rows arrive pre-converted from the encoder.
        model = self._model(["species:a", "species:b"], 4)
        self.assertFalse(hasattr(model, "_remap_category_ids"))


@unittest.skipUnless(torch_available(), "requires torch")
class CompactCheckpointRoundTripTests(unittest.TestCase):
    def test_save_load_preserves_string_vocab(self) -> None:
        from pokezero.neural_policy import (
            EntityTokenTransformerPolicy,
            TransformerEpochMetrics,
            TransformerTrainingConfig,
            TransformerTrainingResult,
            load_transformer_checkpoint,
            save_transformer_checkpoint,
        )

        config = TransformerPolicyConfig.compact_category(
            category_vocab=["move:psychic", "species:blissey", "type:water"], category_oov_buckets=4
        )
        model = EntityTokenTransformerPolicy(config)
        result = TransformerTrainingResult(
            model_config=config,
            training_config=TransformerTrainingConfig(),
            epochs=(TransformerEpochMetrics(epoch=1, examples=1, loss=1.0, policy_loss=1.0, policy_accuracy=0.5),),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compact.pt"
            save_transformer_checkpoint(path, model, result=result)
            loaded, loaded_result = load_transformer_checkpoint(path)
            self.assertEqual(
                loaded_result.model_config.category_vocab, ("move:psychic", "species:blissey", "type:water")
            )
            self.assertEqual(loaded.config.categorical_vocab_size, config.categorical_vocab_size)


if __name__ == "__main__":
    unittest.main()
