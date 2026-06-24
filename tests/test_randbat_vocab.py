import os
from pathlib import Path
import unittest

from pokezero.randbat_vocab import (
    UNIVERSAL_MOVES,
    UNOWN_FORMES,
    build_gen3_randbat_category_vocabulary,
    gen3_randbat_category_strings,
    gen3_randbat_cosmetic_aliases,
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

    def test_universal_moves_present(self) -> None:
        strings = set(gen3_randbat_category_strings(SHOWDOWN_ROOT)["move_action"])
        for move in UNIVERSAL_MOVES:
            self.assertIn(f"move:{move}", strings)

    def test_species_are_display_only_no_dead_id_rows(self) -> None:
        # Punctuation fix: only display-name forms are enumerated, not dead id forms.
        species_strings = set(gen3_randbat_category_strings(SHOWDOWN_ROOT)["species"])
        self.assertIn("species:Mr. Mime", species_strings)
        self.assertNotIn("species:mrmime", species_strings)
        self.assertIn("species:Ho-Oh", species_strings)
        self.assertNotIn("species:hooh", species_strings)

    def test_unown_formes_collapse_to_base_but_deoxys_preserved(self) -> None:
        vocab = set(build_gen3_randbat_category_vocabulary(SHOWDOWN_ROOT))
        species_strings = set(gen3_randbat_category_strings(SHOWDOWN_ROOT)["species"])
        # Cosmetic Unown formes are NOT separate rows; they alias onto base "Unown".
        self.assertIn("species:Unown", species_strings)
        for forme in UNOWN_FORMES:
            self.assertNotIn(f"species:{forme}", species_strings)
        aliases = dict(gen3_randbat_cosmetic_aliases(SHOWDOWN_ROOT))
        base_id = stable_category_id("species:Unown")
        for forme in UNOWN_FORMES:
            self.assertEqual(aliases.get(stable_category_id(f"species:{forme}")), base_id)
        # Functional Deoxys formes are distinct rows and are NOT aliased.
        for forme in ("Deoxys-Attack", "Deoxys-Defense", "Deoxys-Speed"):
            self.assertIn(stable_category_id(f"species:{forme}"), vocab)
            self.assertNotIn(stable_category_id(f"species:{forme}"), aliases)

    def test_known_entities_map_into_vocab(self) -> None:
        vocab = set(build_gen3_randbat_category_vocabulary(SHOWDOWN_ROOT))
        # Display-name forms the encoder emits at play time.
        for value in ("species:Mr. Mime", "move:Aerial Ace", "species:Ho-Oh", "move:Struggle"):
            self.assertIn(stable_category_id(value), vocab)

    def test_no_intra_group_hash_collisions(self) -> None:
        # stable_category_id lowercases+strips, so id/display forms of the same entity share
        # a key (intentional). A true collision is two DISTINCT keys mapping to the same id.
        for name, strings in gen3_randbat_category_strings(SHOWDOWN_ROOT).items():
            keys = {str(s).strip().lower() for s in strings}
            ids = {stable_category_id(key) for key in keys}
            self.assertEqual(len(keys), len(ids), f"hash collision within group {name}")


@unittest.skipUnless(_HAS_GEN3_SETS, "requires a local Pokemon Showdown checkout with gen3 randbat data")
class Gen3RandbatVocabCoverageTests(unittest.TestCase):
    """Live-game coverage: every non-dynamic category the encoder emits must be in the vocab.

    This is the regression guard for enumeration/encoder drift (e.g. request_kind values,
    species/move display forms). Dynamic fields (HP condition text, usernames, free-form
    event details) are intentionally allowed to fall through to the OOV block.
    """

    # Bounded structural/entity prefixes that MUST be covered by the full-universe vocab.
    _REQUIRED_PREFIXES = (
        "species:", "move:", "belief:", "status:", "request_kind:", "pokemon:",
        "event:", "event_actor:", "event_target:", "self_slot:", "opponent_slot:",
        "move_slot:", "switch_slot:",
    )

    def test_live_games_have_no_required_oov(self) -> None:
        try:
            import pokezero.showdown as sd
            from pokezero.collection import BenchmarkMatchup, benchmark_rollouts
            from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
            from pokezero.policy import RandomLegalPolicy
            from pokezero.rollout import RolloutConfig
        except Exception as exc:  # pragma: no cover - environment guard
            self.skipTest(f"runtime deps unavailable: {exc}")

        seen: dict[str, int] = {}
        original = sd.stable_category_id

        def spy(value, *, buckets=sd.CATEGORY_ID_BUCKETS):
            result = original(value, buckets=buckets)
            if buckets == sd.CATEGORY_ID_BUCKETS:
                seen[str(value)] = result
            return result

        sd.stable_category_id = spy
        try:
            benchmark_rollouts(
                games=12,
                env_factory=lambda: LocalShowdownEnv(LocalShowdownConfig(showdown_root=SHOWDOWN_ROOT)),
                rollout_config=RolloutConfig(max_decision_rounds=250),
                seed_start=9100001,
                matchups=[BenchmarkMatchup("r", RandomLegalPolicy(), RandomLegalPolicy())],
            )
        finally:
            sd.stable_category_id = original

        # A required category is "covered" if it is a vocab id OR a cosmetic-forme alias
        # (which the model remaps onto a base-species vocab row).
        covered = set(build_gen3_randbat_category_vocabulary(SHOWDOWN_ROOT))
        covered |= {alias for alias, _ in gen3_randbat_cosmetic_aliases(SHOWDOWN_ROOT)}
        uncovered = [
            value
            for value, cid in seen.items()
            if cid not in covered and value.startswith(self._REQUIRED_PREFIXES)
        ]
        self.assertEqual(uncovered, [], f"required categories fell into OOV: {sorted(uncovered)[:20]}")


if __name__ == "__main__":
    unittest.main()
