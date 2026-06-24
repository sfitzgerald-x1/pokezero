import os
from pathlib import Path
import unittest

from pokezero.randbat_vocab import (
    UNIVERSAL_MOVES,
    UNOWN_FORMES,
    build_gen3_randbat_category_vocabulary,
    gen3_randbat_category_strings,
    gen3_randbat_entities,
    gen3_randbat_vocabulary_breakdown,
)
from pokezero.showdown import stable_category_id

_DEFAULT_SHOWDOWN_ROOT = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
SHOWDOWN_ROOT = os.environ.get("POKEZERO_SHOWDOWN_ROOT", _DEFAULT_SHOWDOWN_ROOT)
_HAS_GEN3_SETS = (Path(SHOWDOWN_ROOT) / "data" / "random-battles" / "gen3" / "sets.json").exists()


@unittest.skipUnless(_HAS_GEN3_SETS, "requires a local Pokemon Showdown checkout with gen3 randbat data")
class Gen3RandbatVocabTests(unittest.TestCase):
    def test_entity_universe_shape(self) -> None:
        entities = gen3_randbat_entities(SHOWDOWN_ROOT)
        self.assertEqual(len(entities["species"]), 220)
        self.assertGreaterEqual(len(entities["moves"]), 120)
        self.assertGreaterEqual(len(entities["abilities"]), 60)
        self.assertTrue(all(isinstance(s, str) for s in entities["species"]))

    def test_vocab_is_sorted_unique_positive(self) -> None:
        vocab = build_gen3_randbat_category_vocabulary(SHOWDOWN_ROOT)
        self.assertEqual(list(vocab), sorted(set(vocab)))
        self.assertTrue(all(v > 0 for v in vocab))

    def test_breakdown_sums_to_distinct_total(self) -> None:
        breakdown = gen3_randbat_vocabulary_breakdown(SHOWDOWN_ROOT)
        total = breakdown.pop("total_distinct")
        vocab = build_gen3_randbat_category_vocabulary(SHOWDOWN_ROOT)
        self.assertEqual(total, len(vocab))
        # No hash collisions across the closed universe: group sizes sum to the total.
        self.assertEqual(sum(breakdown.values()), total)

    def test_universal_moves_and_unown_formes_present(self) -> None:
        strings = set(gen3_randbat_category_strings(SHOWDOWN_ROOT)["move_action"])
        for move in UNIVERSAL_MOVES:
            self.assertIn(f"move:{move}", strings)
        species_strings = set(gen3_randbat_category_strings(SHOWDOWN_ROOT)["species"])
        # Unown is in the gen3 randbat pool, so its cosmetic formes are enumerated.
        self.assertIn(f"species:{UNOWN_FORMES[0]}", species_strings)

    def test_known_entities_map_into_vocab(self) -> None:
        vocab = set(build_gen3_randbat_category_vocabulary(SHOWDOWN_ROOT))
        # Display-name forms the encoder emits at play time.
        for value in ("species:Mr. Mime", "move:Aerial Ace", "species:Ho-Oh", "move:Struggle"):
            self.assertIn(stable_category_id(value), vocab)


if __name__ == "__main__":
    unittest.main()
