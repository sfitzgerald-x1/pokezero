"""Belief-side variant narrowing from external damage evidence.

Covers ``PublicBattleBeliefEngine.narrow_candidate_variants`` — the hook the defender-side
investment inference (``pokezero.investment``) uses to turn a precision-gated stat pin into a
narrowed candidate set, rather than a single reserved observation column.

The behaviour under test is the REFUSAL ASYMMETRY: narrowing is monotone and persists across
re-summarize, but every contradictory or empty signal leaves the standing pin intact. Dropping
the true variant is unrecoverable and would corrupt every belief-derived feature and every
sampled search world; declining a narrowing only costs precision.
"""
import unittest

from pokezero.belief import (
    CandidateSetSummary,
    PublicBattleBeliefEngine,
    RevealedPokemonBelief,
    belief_key,
    variant_identity,
)

# Three Slowbro variants differing in the fields variant_identity reads.
FULL = {"moves": ["surf", "psychic"], "item": "leftovers", "ability": "oblivious", "level": 84}
TRIMMED = {"moves": ["surf", "psychic"], "item": "leftovers", "ability": "owntempo", "level": 84}
OTHER = {"moves": ["surf", "icebeam"], "item": "chestoberry", "ability": "oblivious", "level": 84}
ALL_VARIANTS = (FULL, TRIMMED, OTHER)

KEY = belief_key("p2", "Slowbro")


class VariantSetSource:
    """Returns the full pool every call, mimicking re-derivation on each reveal."""

    def __init__(self, *, inconsistent: bool = False) -> None:
        self.inconsistent = inconsistent

    def summarize(self, *, format_id, species, revealed_moves, **kwargs):
        return CandidateSetSummary(
            species=species,
            candidate_count=len(ALL_VARIANTS),
            # count / species-pool-total, with a pool of 6 -> 0.5
            uncertainty=1.0 if self.inconsistent else len(ALL_VARIANTS) / 6.0,
            candidate_variants=ALL_VARIANTS,
            inconsistent=self.inconsistent,
        )


def _engine(*, inconsistent: bool = False) -> PublicBattleBeliefEngine:
    return PublicBattleBeliefEngine(
        format_id="gen3randombattle", set_source=VariantSetSource(inconsistent=inconsistent)
    )


def _summarized(engine: PublicBattleBeliefEngine) -> RevealedPokemonBelief:
    return engine._with_set_summary(
        RevealedPokemonBelief(showdown_slot="p2", species="Slowbro")
    )


class VariantNarrowingTest(unittest.TestCase):
    def test_identity_discriminates_the_pool(self) -> None:
        self.assertEqual(len({variant_identity(v) for v in ALL_VARIANTS}), 3)

    def test_narrowing_filters_and_rescales_uncertainty(self) -> None:
        engine = _engine()
        self.assertTrue(engine.narrow_candidate_variants(KEY, [FULL], reason="hp-pin"))

        belief = _summarized(engine)
        self.assertEqual(belief.candidate_variants, (FULL,))
        self.assertEqual(belief.candidate_set_count, 1)
        # 3/6 scaled by the surviving fraction 1/3 -> 1/6, i.e. count/pool with pool held fixed.
        self.assertAlmostEqual(belief.uncertainty, 1.0 / 6.0)

    def test_narrowing_survives_resummarize(self) -> None:
        """Each reveal re-derives the set; the pin must outlive that, not be washed out."""
        engine = _engine()
        engine.narrow_candidate_variants(KEY, [FULL, TRIMMED])
        self.assertEqual(len(_summarized(engine).candidate_variants), 2)
        self.assertEqual(len(_summarized(engine).candidate_variants), 2)

    def test_narrowing_is_monotone_intersection(self) -> None:
        engine = _engine()
        engine.narrow_candidate_variants(KEY, [FULL, TRIMMED])
        self.assertTrue(engine.narrow_candidate_variants(KEY, [TRIMMED, OTHER]))
        self.assertEqual(_summarized(engine).candidate_variants, (TRIMMED,))

    def test_a_later_signal_never_widens(self) -> None:
        engine = _engine()
        engine.narrow_candidate_variants(KEY, [FULL])
        self.assertFalse(engine.narrow_candidate_variants(KEY, list(ALL_VARIANTS)))
        self.assertEqual(_summarized(engine).candidate_variants, (FULL,))

    def test_contradictory_signal_keeps_pin_and_counts_conflict(self) -> None:
        engine = _engine()
        engine.narrow_candidate_variants(KEY, [FULL])
        self.assertFalse(engine.narrow_candidate_variants(KEY, [OTHER]))
        self.assertEqual(_summarized(engine).candidate_variants, (FULL,))
        self.assertEqual(engine.variant_pin_conflicts[KEY], 1)

    def test_empty_survivors_are_not_evidence(self) -> None:
        engine = _engine()
        self.assertFalse(engine.narrow_candidate_variants(KEY, []))
        self.assertEqual(len(_summarized(engine).candidate_variants), 3)

    def test_inconsistent_summary_is_never_narrowed(self) -> None:
        """A fallback pool with uncertainty forced to 1.0 must not be made to look confident."""
        engine = _engine(inconsistent=True)
        engine.narrow_candidate_variants(KEY, [FULL])
        belief = _summarized(engine)
        self.assertEqual(len(belief.candidate_variants), 3)
        self.assertAlmostEqual(belief.uncertainty, 1.0)

    def test_clone_carries_the_pin(self) -> None:
        """A sampled world that forgot the pin would re-admit excluded variants."""
        engine = _engine()
        engine.narrow_candidate_variants(KEY, [FULL])
        self.assertEqual(_summarized(engine.clone()).candidate_variants, (FULL,))

    def test_unpinned_key_is_untouched(self) -> None:
        engine = _engine()
        engine.narrow_candidate_variants(belief_key("p2", "Snorlax"), [FULL])
        self.assertEqual(len(_summarized(engine).candidate_variants), 3)


if __name__ == "__main__":
    unittest.main()
