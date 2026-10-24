"""Exact input identity fixtures, not a native leaf-parity certificate."""
from dataclasses import replace
import struct
import unittest
from unittest.mock import patch

from pokezero.shallow_exact import ExactSearchRefused
from pokezero.shallow_leaf_rows import DecisionLeafValues, EncodedLeafRow, encoded_leaf_row


class EncodedLeafRowsTests(unittest.TestCase):
    def setUp(self):
        self.row = {"categorical_ids": struct.pack("<ii", 3, 5),
                    "numeric_features": struct.pack("<dd", 0.5, 0.75),
                    "token_type_ids": struct.pack("<h", 2),
                    "attention_mask": b"\x01", "legal_action_mask": b"\x01\x00"}
        self.options = dict(model_sha256="a" * 64, window=1, tokens=1,
                            categorical_features=2, numeric_features=2)

    def key(self, row=None, **options):
        return encoded_leaf_row(self.row if row is None else row, **(self.options | options))

    def test_actual_forward_dtypes_shapes_and_five_arrays(self):
        key = self.key()
        self.assertEqual(key.flat_inputs(), ([3, 5], [0.5, 0.75], [2], [True], [True]))
        self.assertEqual(tuple((dtype, shape) for dtype, shape, _ in key.arrays),
                         (("<q", (1, 1, 2)), ("<f", (1, 1, 2)), ("<q", (1, 1)),
                          ("<?", (1, 1)), ("<?", (1,))))
        self.assertEqual(key, self.key(dict(self.row)))

    def test_every_array_legal_mask_shape_and_model_remain_distinct(self):
        first = self.key()
        for name, data in (("categorical_ids", struct.pack("<ii", 3, 6)),
                           ("numeric_features", struct.pack("<dd", 0.5, 0.8)),
                           ("token_type_ids", struct.pack("<h", 3)),
                           ("attention_mask", b"\x00"), ("legal_action_mask", b"\x00\x01")):
            with self.subTest(name=name):
                self.assertNotEqual(first, self.key(self.row | {name: data}))
        self.assertNotEqual(first, self.key(model_sha256="b" * 64))
        changed = list(first.arrays)
        dtype, shape, data = changed[4]
        changed[4] = (dtype, shape, b"\x00")
        self.assertNotEqual(first, replace(first, arrays=tuple(changed)))
        changed[0] = ("<i", (1, 1, 2), changed[0][2])
        self.assertNotEqual(first, replace(first, arrays=tuple(changed)))
        changed[0] = ("<q", (1, 2, 1), changed[0][2])
        self.assertNotEqual(first, replace(first, arrays=tuple(changed)))

    def test_hash_collision_cannot_merge_non_equivalent_rows(self):
        first = self.key()
        second = self.key(self.row | {"token_type_ids": struct.pack("<h", 8)})
        with patch.object(EncodedLeafRow, "__hash__", return_value=0):
            cache = {first: 0.25, second: -0.75}
            self.assertEqual(len(cache), 2)
            self.assertEqual((cache[first], cache[second]), (0.25, -0.75))

    def test_only_effectively_identical_float32_values_may_coalesce(self):
        self.assertEqual(self.key(), self.key(self.row | {
            "numeric_features": struct.pack("<dd", 0.5 + 1e-10, 0.75)}))
        self.assertNotEqual(self.key(), self.key(self.row | {
            "numeric_features": struct.pack("<dd", 0.5 + 1e-6, 0.75)}))

    def test_malformed_rows_refuse_before_forward(self):
        for name, data in (("categorical_ids", b"x"), ("token_type_ids", b"x"),
                           ("attention_mask", b"\x02"), ("legal_action_mask", b"\x02"),
                           ("numeric_features", struct.pack("<dd", float("nan"), 0.5)),
                           ("numeric_features", struct.pack("<dd", 1e100, 0.5))):
            with self.subTest(name=name, data=data):
                with self.assertRaises(ExactSearchRefused):
                    self.key(self.row | {name: data})
        with self.assertRaises(ExactSearchRefused):
            self.key(model_sha256="unbound")
        with self.assertRaises(ExactSearchRefused):
            self.key(tokens=2)

    def test_unique_forward_scatter_and_per_decision_counter_reset(self):
        class Model:
            def __init__(self):
                self.calls = []
            def eval_obs_flat(self, count, *arrays):
                self.calls.append((count, arrays))
                return [arrays[1][2 * i] for i in range(count)], [], [], 0
        model = Model()
        memo = DecisionLeafValues(model, model_sha256="a" * 64)
        first = self.key()
        second = self.key(self.row | {"numeric_features": struct.pack("<dd", 0.25, 0.75)})
        with patch.object(EncodedLeafRow, "__hash__", return_value=0):
            self.assertEqual(memo.evaluate([first, second, first], model), [0.5, 0.25, 0.5])
            self.assertEqual(memo.evaluate([second, first], model), [0.25, 0.5])
        self.assertEqual((memo.row_requests, memo.cache_hits, memo.unique_evaluations,
                          memo.forward_batches, len(model.calls)), (5, 3, 2, 1, 1))
        self.assertEqual(model.calls[0][0], 2)
        memo.begin_decision(model)
        self.assertEqual(memo.evaluate([first], model), [0.5])
        self.assertEqual((memo.row_requests, memo.cache_hits, memo.unique_evaluations,
                          memo.forward_batches), (1, 0, 1, 1))
        self.assertEqual(len(model.calls), 2)

    def test_changed_model_or_shape_cannot_reuse_cached_value(self):
        class Model:
            def eval_obs_flat(self, count, *arrays):
                return [0.25] * count, [], [], 0
        model = Model()
        memo = DecisionLeafValues(model, model_sha256="a" * 64)
        self.assertEqual(memo.evaluate([self.key()], model), [0.25])
        for rows, supplied_model in (([self.key()], Model()),
                                     ([self.key(model_sha256="b" * 64)], model),
                                     ([replace(self.key(), arrays=self.key().arrays[:-1])], model)):
            with self.subTest(rows=rows):
                with self.assertRaises(ExactSearchRefused):
                    memo.evaluate(rows, supplied_model)
        with self.assertRaises(ExactSearchRefused):
            memo.begin_decision(Model())

    def test_failed_batch_does_not_publish_partial_values(self):
        class Model:
            def eval_obs_flat(self, count, *arrays):
                return [float("nan")] * count, [], [], 0
        model = Model()
        memo = DecisionLeafValues(model, model_sha256="a" * 64)
        with self.assertRaisesRegex(ExactSearchRefused, "finite"):
            memo.evaluate([self.key()], model)
        self.assertEqual(memo.values, {})
        self.assertEqual(memo.unique_evaluations, 0)

    def test_returned_cardinality_must_match_unique_rows(self):
        class Model:
            def eval_obs_flat(self, count, *arrays):
                return [], [], [], 0
        model = Model()
        memo = DecisionLeafValues(model, model_sha256="a" * 64)
        with self.assertRaisesRegex(ExactSearchRefused, "cardinality"):
            memo.evaluate([self.key()], model)
        self.assertEqual(memo.values, {})


if __name__ == "__main__":
    unittest.main()
