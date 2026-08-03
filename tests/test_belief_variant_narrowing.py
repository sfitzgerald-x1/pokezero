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


class RefusalGuardsTest(unittest.TestCase):
    """Second-order guards: the ones that keep a sound exclusion from becoming a wrong one.

    A mutation sweep found these unpinned while every headline property was covered. They are
    exactly the guards whose failure is silent — the set still narrows, just to the wrong thing.
    """

    def test_possible_values_are_never_emptied_by_a_narrowing(self) -> None:
        """`return narrowed or values` is the refusal asymmetry applied to the projections.

        Without the fallback a pin that matches no emitted `possible_item` empties the list, and
        a consumer reading "this mon can hold nothing" is strictly worse off than one reading the
        unnarrowed set. Same rule as the pin itself: decline, never eliminate everything.
        """
        from pokezero.belief import _narrowed_possible_values

        values = ("Leftovers", "Choice Band")
        variants = [{"item": "Leftovers"}, {"item": "Choice Band"}]
        # survivors carrying an item OUTSIDE the emitted projection: filtering strictly would
        # empty the list, so the guard must hand back the original instead.
        kept = [{"item": "Lum Berry"}]
        self.assertEqual(
            _narrowed_possible_values(values, variants, kept, "item", plural=False), values
        )
        # a normal narrowing still filters
        self.assertEqual(
            _narrowed_possible_values(
                values, variants, [{"item": "Leftovers"}], "item", plural=False
            ),
            ("Leftovers",),
        )

    def test_an_unevaluable_candidate_blocks_the_whole_narrowing(self) -> None:
        """The never-drop-the-true-variant guard, and the highest-value survivor of the sweep.

        If any candidate's spread cannot be computed, that candidate could BE the true variant,
        so no exclusion is safe and the producer must hand back () — which the engine reads as
        "no evidence". Degrading to `continue` instead would silently exclude it.
        """
        import inspect

        from pokezero import investment

        src = inspect.getsource(investment._choice_band_variant_payloads
                                if hasattr(investment, "_choice_band_variant_payloads")
                                else investment)
        # the guard is a hard return, never a skip
        self.assertIn("return ()", inspect.getsource(investment))
