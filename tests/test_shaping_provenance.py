"""Shaping provenance: cache stamps, checkpoint stamps, and the train cross-check."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.collection import cache_shaping_configs_by_path
from pokezero.shaping import SHAPING_PRESETS

WSE = SHAPING_PRESETS["wse-arm1"]


def _write_cache_metadata(path: Path, dataset_config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps({"schema_version": "pokezero.training_cache.v2", "dataset_config": dataset_config}),
        encoding="utf-8",
    )


class CacheShapingProvenanceTest(unittest.TestCase):
    def test_cache_shaping_configs_by_path_classifies_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shaped = root / "shaped"
            unshaped = root / "unshaped"
            broken = root / "broken"
            _write_cache_metadata(shaped, {"window_size": 1, "potential_shaping": WSE.to_dict()})
            _write_cache_metadata(unshaped, {"window_size": 1})
            broken.mkdir()
            (broken / "metadata.json").write_text("{not json", encoding="utf-8")
            jsonl = root / "records.jsonl"
            jsonl.write_text("", encoding="utf-8")

            rows = cache_shaping_configs_by_path([shaped, unshaped, broken, jsonl, root / "missing"])

        by_path = {path: (shaping, checkable) for path, shaping, checkable in rows}
        self.assertEqual(by_path[shaped], (WSE.to_dict(), True))
        self.assertEqual(by_path[unshaped], (None, True))
        self.assertEqual(by_path[broken], (None, False))
        # JSONL and missing paths are omitted entirely (mask-helper convention).
        self.assertEqual(set(by_path), {shaped, unshaped, broken})


class TrainCrossCheckTest(unittest.TestCase):
    def _check(self, paths, shaping_json):
        from pokezero.neural_cli import _require_cache_shaping_matches_training_config

        _require_cache_shaping_matches_training_config(paths, shaping_json)

    def test_cross_check_passes_on_agreement_and_fails_both_directions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shaped = root / "shaped"
            unshaped = root / "unshaped"
            _write_cache_metadata(shaped, {"window_size": 1, "potential_shaping": WSE.to_dict()})
            _write_cache_metadata(unshaped, {"window_size": 1})

            self._check([shaped], WSE.canonical_json())  # no raise
            self._check([unshaped], None)  # no raise
            with self.assertRaisesRegex(ValueError, "shaping"):
                self._check([shaped], None)  # shaped cache, unshaped run
            with self.assertRaisesRegex(ValueError, "shaping"):
                self._check([unshaped], WSE.canonical_json())  # unshaped cache, shaped run

    def test_cross_check_skips_unreadable_metadata_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            broken = root / "broken"
            broken.mkdir()
            (broken / "metadata.json").write_text("{not json", encoding="utf-8")
            jsonl = root / "records.jsonl"
            jsonl.write_text("", encoding="utf-8")

            self._check([broken, jsonl], WSE.canonical_json())  # no raise
            self._check([broken, jsonl], None)  # no raise


class CheckpointShapingStampTest(unittest.TestCase):
    def _model_config(self, **kwargs):
        from pokezero.neural_policy import TransformerPolicyConfig

        return TransformerPolicyConfig.compact_category(
            category_vocab=("alakazam", "zapdos"), category_oov_buckets=1, **kwargs
        )

    def test_model_config_normalizes_and_round_trips_reward_shaping(self) -> None:
        config = self._model_config(reward_shaping="wse-arm1")
        self.assertEqual(config.reward_shaping, WSE.canonical_json())
        payload = config.to_dict()
        self.assertEqual(payload["reward_shaping"], WSE.canonical_json())
        from pokezero.neural_policy import TransformerPolicyConfig

        self.assertEqual(TransformerPolicyConfig.from_dict(payload), config)

    def test_from_dict_defaults_missing_field_to_unshaped(self) -> None:
        from pokezero.neural_policy import TransformerPolicyConfig

        payload = self._model_config().to_dict()
        payload.pop("reward_shaping", None)
        self.assertIsNone(TransformerPolicyConfig.from_dict(payload).reward_shaping)

    def test_training_config_normalizes_shaping_and_resolves(self) -> None:
        from pokezero.neural_policy import TransformerTrainingConfig

        config = TransformerTrainingConfig(shaping_weights='{"hp_weight": 0.5, "status_weight": 0.25}')
        resolved = config.resolved_shaping_config()
        self.assertEqual(resolved.hp_weight, 0.5)
        self.assertEqual(resolved.status_weight("slp"), 0.25)
        self.assertEqual(config.shaping_weights, resolved.canonical_json())
        self.assertIsNone(TransformerTrainingConfig().resolved_shaping_config())
        with self.assertRaisesRegex(ValueError, "shaping_weights"):
            TransformerTrainingConfig(shaping_weights="none")
        # Checkpoint payload round-trip via TransformerTrainingConfig(**payload).
        self.assertEqual(
            TransformerTrainingConfig(**config.to_dict()).shaping_weights, config.shaping_weights
        )

    def test_resolved_training_shaping_adopts_and_retargets(self) -> None:
        from types import SimpleNamespace

        from pokezero.neural_cli import _resolved_training_shaping_json

        checkpoint = SimpleNamespace(model_config=SimpleNamespace(reward_shaping=WSE.canonical_json()))
        unshaped_checkpoint = SimpleNamespace(model_config=SimpleNamespace(reward_shaping=None))
        absent = argparse.Namespace(shaping_weights=None)
        explicit_on = argparse.Namespace(shaping_weights="wse-arm1")
        explicit_off = argparse.Namespace(shaping_weights="none")

        self.assertIsNone(_resolved_training_shaping_json(absent, None))
        self.assertEqual(_resolved_training_shaping_json(explicit_on, None), WSE.canonical_json())
        # Resume without the flag inherits the checkpoint stamp.
        self.assertEqual(_resolved_training_shaping_json(absent, checkpoint), WSE.canonical_json())
        self.assertIsNone(_resolved_training_shaping_json(absent, unshaped_checkpoint))
        # Explicit flag re-targets in either direction.
        self.assertIsNone(_resolved_training_shaping_json(explicit_off, checkpoint))
        self.assertEqual(
            _resolved_training_shaping_json(explicit_on, unshaped_checkpoint), WSE.canonical_json()
        )


if __name__ == "__main__":
    unittest.main()
