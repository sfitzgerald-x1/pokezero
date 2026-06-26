import unittest

from pokezero.category_vocab import CategoryVocabulary, build_category_vocabulary


class CategoryVocabularyTest(unittest.TestCase):
    def _vocab(self) -> CategoryVocabulary:
        return build_category_vocabulary(
            ["species:Charizard", "move:Psychic", "type:fire"],
            oov_buckets=16,
            aliases={"species:Charizard-Mega": "species:Charizard"},
        )

    def test_size_and_rows(self) -> None:
        v = self._vocab()
        self.assertEqual(v.size, 1 + 3 + 16)
        # sorted: move:psychic(1), species:charizard(2), type:fire(3)
        self.assertEqual(v.encode("move:Psychic"), 1)
        self.assertEqual(v.encode("species:Charizard"), 2)
        self.assertEqual(v.encode("type:fire"), 3)

    def test_normalization(self) -> None:
        v = self._vocab()
        self.assertEqual(v.encode("SPECIES:Charizard"), v.encode("species:charizard"))
        self.assertEqual(v.encode("  type:FIRE  "), 3)

    def test_padding_and_alias(self) -> None:
        v = self._vocab()
        self.assertEqual(v.encode(""), 0)
        self.assertEqual(v.encode(None), 0)
        self.assertEqual(v.encode("species:Charizard-Mega"), v.encode("species:Charizard"))

    def test_oov_in_reserved_band(self) -> None:
        v = self._vocab()
        row = v.encode("species:Missingno")
        self.assertGreaterEqual(row, 1 + 3)  # at/after the OOV offset
        self.assertLess(row, v.size)

    def test_oov_logs_drift_once_and_is_recorded(self) -> None:
        v = self._vocab()
        self.assertEqual(v.observed_oov_tokens, frozenset())  # healthy: nothing hashed yet
        with self.assertLogs("pokezero.category_vocab", level="WARNING") as captured:
            v.encode("species:Missingno")
            v.encode("species:Missingno")  # repeat must not warn again
            v.encode("MOVE:Frobnicate")  # a second distinct OOV token warns
        self.assertEqual(len(captured.records), 2)  # warn-once per distinct token
        self.assertEqual(
            v.observed_oov_tokens, frozenset({"species:missingno", "move:frobnicate"})
        )

    def test_in_vocabulary_tokens_do_not_log_drift(self) -> None:
        v = self._vocab()
        with self.assertNoLogs("pokezero.category_vocab", level="WARNING"):
            v.encode("species:Charizard")
            v.encode("species:Charizard-Mega")  # alias resolves to a real row, not OOV
            v.encode("")  # padding
        self.assertEqual(v.observed_oov_tokens, frozenset())

    def test_rejects_duplicate_after_normalization(self) -> None:
        with self.assertRaises(ValueError):
            CategoryVocabulary(tokens=("move:psychic", "MOVE:Psychic"))


if __name__ == "__main__":
    unittest.main()
