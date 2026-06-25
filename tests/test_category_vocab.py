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

    def test_rejects_duplicate_after_normalization(self) -> None:
        with self.assertRaises(ValueError):
            CategoryVocabulary(tokens=("move:psychic", "MOVE:Psychic"))


if __name__ == "__main__":
    unittest.main()
