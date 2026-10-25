"""Search targets survive the actual collector-to-cache paths, without changing PPO data."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from pokezero.collection import write_rollout_record
from pokezero.dataset import (
    TrajectoryDatasetConfig, TrainingCacheBuilder, _combine_training_batches,
    concat_training_caches, examples_from_record, iter_training_batches,
    iter_training_cache_batches, slice_training_batch, training_batch_from_examples,
    write_training_cache_from_examples, write_training_cache_streaming,
)
from pokezero.search_label_data import SEARCH_COLUMNS, SEARCH_PROVENANCE
from tests.test_dataset import rollout_record


POLICY = (1.0,) + (0.0,) * 8
TEACHER = {"checkpoint_sha256": "a" * 64, "iteration": 25, "export_sha256": "b" * 64}
LABELED = {"status": "LABELED", "teacher": TEACHER,
           "policy_target": list(POLICY), "value_target": -0.25}
REFUSED = {"status": "UNLABELED", "teacher": TEACHER,
           "reason": "teacher_fallback", "refusal_code": "fallback.crate_search_failed"}


def record_with_labels():
    r = rollout_record()
    r.trajectory.steps[0] = replace(r.trajectory.steps[0], metadata={"search_label": LABELED})
    r.trajectory.steps[1] = replace(r.trajectory.steps[1], metadata={"search_label": REFUSED})
    return r


@unittest.skipIf(np is None, "requires optional NumPy dependency")
class SearchLabelDatasetTest(unittest.TestCase):
    def assert_labels(self, batch, offset=0):
        np.testing.assert_allclose(batch.search_policy_targets[offset], POLICY)
        self.assertAlmostEqual(batch.search_value_targets[offset], -0.25)
        self.assertEqual(list(batch.search_policy_target_mask)[offset:offset + 3], [True, False, False])
        self.assertEqual(list(batch.search_value_target_mask)[offset:offset + 3], [True, False, False])
        self.assertEqual(batch.step_metadata[offset]["search_label"]["teacher"], TEACHER)
        self.assertEqual(batch.step_metadata[offset + 1]["search_label"], REFUSED)

    def test_record_to_examples_preserves_behavior_and_separate_targets(self):
        cfg = TrajectoryDatasetConfig(window_size=2, ppo_target_mode="gae")
        old = list(examples_from_record(rollout_record(), config=cfg))
        new = list(examples_from_record(record_with_labels(), config=cfg))
        for a, b in zip(old, new, strict=True):
            for field in ("action_index", "action_probability", "value_estimate", "reward", "return_value",
                          "ppo_advantage", "ppo_value_target", "categorical_ids", "legal_action_mask"):
                self.assertEqual(getattr(a, field), getattr(b, field), field)
        self.assertEqual(new[0].search_policy_target, POLICY)
        self.assertEqual(new[0].search_value_target, -0.25)
        self.assertIsNone(new[1].search_value_target)
        self.assert_labels(training_batch_from_examples(new))

    def test_both_cache_writers_match_bytes_and_preserve_attempts(self):
        cfg = TrajectoryDatasetConfig(window_size=2)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); source = root / "rollouts.jsonl"
            with source.open("w") as out:
                write_rollout_record(out, record_with_labels())
            builder = TrainingCacheBuilder(config=cfg); builder.add_record(record_with_labels())
            builder.write(root / "memory", max_cache_root_bytes=None)
            write_training_cache_streaming(source, root / "stream", config=cfg,
                                           max_cache_root_bytes=None, flush_rows=1)
            for f in (root / "memory").iterdir():
                self.assertEqual(f.read_bytes(), (root / "stream" / f.name).read_bytes(), f.name)
            for deferred in (False, True):
                for cache in ("memory", "stream"):
                    b = next(iter_training_cache_batches(root / cache, batch_size=3, defer_window_expansion=deferred))
                    self.assert_labels(b)
                    sliced = slice_training_batch(b, 1, 3)
                    self.assertEqual(list(sliced.search_policy_target_mask), [False, False])
                    self.assertEqual(sliced.step_metadata[0]["search_label"], REFUSED)

    def test_example_writer_retains_independent_masks(self):
        examples = list(examples_from_record(rollout_record()))
        examples[0] = replace(examples[0], search_policy_target=POLICY)
        examples[1] = replace(examples[1], search_value_target=0.75)
        with tempfile.TemporaryDirectory() as t:
            cache = Path(t) / "cache"
            write_training_cache_from_examples(examples, cache, max_cache_root_bytes=None)
            b = next(iter_training_cache_batches(cache, batch_size=3))
            self.assertEqual(list(b.search_policy_target_mask), [True, False, False])
            self.assertEqual(list(b.search_value_target_mask), [False, True, False])
            self.assertAlmostEqual(b.search_value_targets[1], 0.75)

    def test_mixed_old_new_concat_matches_single_cache_in_both_orders(self):
        for labeled_first in (False, True):
            with self.subTest(labeled_first=labeled_first), tempfile.TemporaryDirectory() as t:
                root = Path(t); records = [rollout_record(), record_with_labels()]
                if labeled_first:
                    records.reverse()
                for i, r in enumerate(records):
                    builder = TrainingCacheBuilder(); builder.add_record(r)
                    builder.write(root / str(i), max_cache_root_bytes=None)
                concat_training_caches([root / "0", root / "1"], root / "joined")
                builder = TrainingCacheBuilder()
                for r in records:
                    builder.add_record(r)
                builder.write(root / "single", max_cache_root_bytes=None)
                for f in (root / "single").iterdir():
                    self.assertEqual(f.read_bytes(), (root / "joined" / f.name).read_bytes(), f.name)
                batch = next(iter_training_cache_batches(root / "joined", batch_size=6))
                self.assert_labels(batch, offset=0 if labeled_first else 3)

    def test_mixed_cache_coalescing_and_direct_batch_combine(self):
        for deferred in (False, True):
            with self.subTest(deferred=deferred), tempfile.TemporaryDirectory() as t:
                root = Path(t)
                for name, r in (("old", rollout_record()), ("new", record_with_labels())):
                    builder = TrainingCacheBuilder(); builder.add_record(r)
                    builder.write(root / name, max_cache_root_bytes=None)
                batch = next(iter_training_batches([root / "old", root / "new"], batch_size=6,
                                                   defer_cache_window_expansion=deferred))
                self.assert_labels(batch, offset=3)
                self.assertEqual(list(batch.search_policy_target_mask)[:3], [False] * 3)
        old = training_batch_from_examples(list(examples_from_record(rollout_record())))
        new = training_batch_from_examples(list(examples_from_record(record_with_labels())))
        self.assert_labels(_combine_training_batches([old, new]), offset=3)
        self.assert_labels(_combine_training_batches([new, old]))

    def test_off_cache_emits_no_new_columns_or_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            cache = Path(t) / "cache"; builder = TrainingCacheBuilder(); builder.add_record(rollout_record())
            builder.write(cache, max_cache_root_bytes=None)
            meta = json.loads((cache / "metadata.json").read_text())
            self.assertNotIn(SEARCH_PROVENANCE, meta)
            for n in SEARCH_COLUMNS:
                self.assertFalse((cache / (n + ".npy")).exists())
                self.assertNotIn(n, meta["array_dtypes"])
            b = next(iter_training_cache_batches(cache, batch_size=3))
            self.assertIsNone(b.search_policy_targets)
            self.assertEqual(b.step_metadata, ({}, {}, {}))

    def test_invalid_or_unlabeled_target_is_not_silently_admitted(self):
        for label in ({**LABELED, "value_target": float("nan")},
                      {**LABELED, "policy_target": [0.0, 1.0] + [0.0] * 7},
                      {**REFUSED, "policy_target": list(POLICY)},
                      {"status": "UNKNOWN"}):
            with self.subTest(label=label), self.assertRaises(ValueError):
                r = rollout_record(); r.trajectory.steps[0] = replace(r.trajectory.steps[0], metadata={"search_label": label})
                list(examples_from_record(r))

    def test_partial_cache_column_bundle_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            cache = Path(t) / "cache"; builder = TrainingCacheBuilder(); builder.add_record(rollout_record())
            builder.write(cache, max_cache_root_bytes=None)
            np.save(cache / "search_value_targets.npy", np.zeros(3))
            with self.assertRaisesRegex(ValueError, "present together"):
                next(iter_training_cache_batches(cache, batch_size=3))

    def test_torch_conversion_preserves_targets_and_omits_off_columns(self):
        from pokezero.neural_policy import training_batch_to_torch, torch_available
        if not torch_available():
            self.skipTest("requires optional PyTorch dependency")
        import torch

        old = training_batch_from_examples(list(examples_from_record(rollout_record())))
        self.assertTrue(set(SEARCH_COLUMNS).isdisjoint(training_batch_to_torch(old)))
        new = training_batch_from_examples(list(examples_from_record(record_with_labels())))
        tensors = training_batch_to_torch(new)
        self.assertEqual(tensors["search_policy_targets"].dtype, torch.float32)
        self.assertEqual(tensors["search_policy_target_mask"].dtype, torch.bool)
        self.assertEqual(tensors["search_policy_targets"].shape, (3, 9))
        self.assertEqual(tensors["search_policy_target_mask"].tolist(), [True, False, False])
        self.assertEqual(tensors["search_value_targets"].tolist(), [-0.25, 0.0, 0.0])

    def test_declared_labels_cannot_silently_become_legacy_in_load_or_concat(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); cache = root / "cache"
            builder = TrainingCacheBuilder(); builder.add_record(record_with_labels())
            builder.write(cache, max_cache_root_bytes=None)
            for n in SEARCH_COLUMNS:
                (cache / (n + ".npy")).unlink()
            with self.assertRaisesRegex(ValueError, "missing target columns"):
                next(iter_training_cache_batches(cache, batch_size=3))
            with self.assertRaisesRegex(ValueError, "missing target columns"):
                concat_training_caches([cache], root / "joined")

    def test_malformed_shapes_and_non_boolean_masks_refuse(self):
        batch = training_batch_from_examples(list(examples_from_record(record_with_labels())))
        for field, value in (("search_policy_targets", np.zeros((3, 9, 1))),
                             ("search_value_targets", np.zeros((3, 1))),
                             ("search_value_target_mask", np.zeros((3, 1), dtype=bool)),
                             ("search_policy_target_mask", [1, 0, 0]),
                             ("search_policy_target_mask", np.ones(3, dtype=float))):
            with self.subTest(field=field, value=str(value)), self.assertRaises(ValueError):
                replace(batch, **{field: value})

    def test_corrupt_provenance_refuses_in_load_and_concat(self):
        for mutation in ("outside_index", "unlabeled_mask", "missing_teacher", "labeled_mask", "changed_target"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as t:
                root = Path(t); cache = root / "cache"
                builder = TrainingCacheBuilder(); builder.add_record(record_with_labels())
                builder.write(cache, max_cache_root_bytes=None)
                meta = json.loads((cache / "metadata.json").read_text())
                if mutation == "outside_index":
                    meta[SEARCH_PROVENANCE]["4"] = meta[SEARCH_PROVENANCE].pop("0")
                elif mutation == "unlabeled_mask":
                    mask = np.array([True, True, False]); np.save(cache / "search_policy_target_mask.npy", mask)
                elif mutation == "missing_teacher":
                    del meta[SEARCH_PROVENANCE]["0"]["search_label"]["teacher"]
                elif mutation == "labeled_mask":
                    np.save(cache / "search_value_target_mask.npy", np.zeros(3, dtype=bool))
                else:
                    np.save(cache / "search_value_targets.npy", np.zeros(3))
                (cache / "metadata.json").write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    next(iter_training_cache_batches(cache, batch_size=3))
                with self.assertRaises(ValueError):
                    concat_training_caches([cache], root / "joined")


if __name__ == "__main__":
    unittest.main()
