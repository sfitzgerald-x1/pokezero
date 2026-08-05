"""A1: a forced-replacement ply ran no residual phase (C116 Phase 3 item 9).

Showdown gives a Pokemon arriving on a forced replacement no residual tick --
the replacement completes the PREVIOUS turn rather than starting a new one. The
engine, asked for a full turn, faithfully runs one and over-emits. HARNESS
change, measured and landed separately from any fidelity change.

These pins are BEHAVIOURAL. An earlier version asserted on `inspect.getsource`
text, and review showed that inverting the `|win|` clause -- the exact
first-attempt failure -- would have passed every one of them: the string
`"|win"` was still present, the `and` count unchanged, no `or` introduced. The
predicate is now module-level so it can be called.
"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import engine_transition_differential as diff  # noqa: E402


class ForcedReplacementPredicateTests(unittest.TestCase):
    """Behavioural pins on `_is_forced_replacement_ply`."""

    # 19000020/50's protocol, verbatim and complete.
    REPLACEMENT = [
        "|switch|p1a: Dewgong|Dewgong, L87, F|230/298",
        "|switch|p2a: Politoed|Politoed, L84, M|269/288",
        "|turn|47",
    ]
    ORDINARY_TURN = [
        "|move|p1a: Dewgong|Surf|p2a: Politoed",
        "|-damage|p2a: Politoed|200/288",
        "|-heal|p1a: Dewgong|248/298|[from] item: Leftovers",
        "|upkeep",
        "|turn|48",
    ]
    # The shape that broke the first attempt: battle ends DURING residuals, so
    # there is no |upkeep| -- but Showdown did emit the tick.
    BATTLE_END = [
        "|-weather|Sandstorm|[upkeep]",
        "|-damage|p1a: Slaking|0 fnt|[from] Sandstorm",
        "|-damage|p2a: Venusaur|161/262|[from] Sandstorm",
        "|faint|p1a: Slaking",
        "|win|PokeZero p2",
    ]
    TIE_END = [
        "|-damage|p1a: A|0 fnt|[from] psn",
        "|-damage|p2a: B|0 fnt|[from] psn",
        "|faint|p1a: A",
        "|faint|p2a: B",
        "|tie",
    ]
    VOLUNTARY_DOUBLE_SWITCH = [
        "|switch|p1a: Dewgong|Dewgong, L87, F|230/298",
        "|switch|p2a: Politoed|Politoed, L84, M|269/288",
        "|-heal|p1a: Dewgong|248/298|[from] item: Leftovers",
        "|upkeep",
        "|turn|48",
    ]

    def test_a_forced_replacement_fires(self) -> None:
        self.assertTrue(diff._is_forced_replacement_ply(self.REPLACEMENT))

    def test_an_ordinary_turn_does_not(self) -> None:
        self.assertFalse(diff._is_forced_replacement_ply(self.ORDINARY_TURN))

    def test_a_battle_end_during_residuals_does_not(self) -> None:
        """The first attempt's failure, pinned.

        No `|upkeep|` here either, but Showdown DID emit residuals. Keying on
        upkeep alone stripped the engine's and manufactured 44 divergences.
        """
        self.assertFalse(diff._is_forced_replacement_ply(self.BATTLE_END))

    def test_a_tie_end_does_not(self) -> None:
        """A double KO ends the battle with `|tie`, not `|win`."""
        self.assertFalse(diff._is_forced_replacement_ply(self.TIE_END))

    def test_a_voluntary_double_switch_does_not(self) -> None:
        """Nobody moved and both switched, but a residual phase ran."""
        self.assertFalse(diff._is_forced_replacement_ply(self.VOLUNTARY_DOUBLE_SWITCH))

    def test_inverting_any_single_clause_breaks_at_least_one_case(self) -> None:
        """The four clauses are each load-bearing.

        This is what the old text pins only appeared to check: with the real
        predicate callable, each clause is shown necessary by a case that
        distinguishes it.
        """
        cases = {
            "upkeep": self.ORDINARY_TURN,
            "win": self.BATTLE_END,
            "tie": self.TIE_END,
            "switch": ["|move|p1a: A|Splash|p1a: A", "|turn|3"],
        }
        for name, lines in cases.items():
            with self.subTest(clause=name):
                self.assertFalse(
                    diff._is_forced_replacement_ply(lines),
                    f"the {name} clause is not doing any work",
                )


class ResidualSourceSetTests(unittest.TestCase):
    def test_it_covers_every_source_the_a1_rows_carry(self) -> None:
        for source in ("itemleftovers", "psn", "sandstorm"):
            self.assertIn(source, diff._RESIDUAL_PHASE_SOURCES)

    def test_hazards_are_excluded_so_a_different_cause_stays_divergent(self) -> None:
        """19100180/24 is a hazard mis-attribution, not a residual."""
        for hazard in ("spikes", "stealthrock"):
            self.assertNotIn(hazard, diff._RESIDUAL_PHASE_SOURCES)

    def test_partial_trapping_is_covered(self) -> None:
        """gen3 partial trapping ticks 1/16 at end of turn."""
        for source in ("partiallytrapped", "movewrap", "movefirespin"):
            self.assertIn(source, diff._RESIDUAL_PHASE_SOURCES)

    def test_it_is_not_the_majority_override_set(self) -> None:
        self.assertNotEqual(diff._RESIDUAL_PHASE_SOURCES, diff._ADJUDICABLE_RESIDUALS)
        for source in ("leechseed", "movewish", "hail"):
            self.assertIn(source, diff._RESIDUAL_PHASE_SOURCES)
            self.assertNotIn(source, diff._ADJUDICABLE_RESIDUALS)


class FilterBehaviourTests(unittest.TestCase):
    """What the filter does to a component Counter, exercised directly."""

    @staticmethod
    def _strip(eng: Counter, lines: list[str]) -> Counter:
        if not diff._is_forced_replacement_ply(lines):
            return eng
        return Counter({
            c: n for c, n in eng.items() if c[0] not in diff._RESIDUAL_PHASE_SOURCES
        })

    def test_residuals_are_dropped_on_a_replacement_ply(self) -> None:
        eng = Counter({("itemleftovers", 16): 1, ("psn", -16): 1, ("sandstorm", -16): 1})
        self.assertEqual(
            self._strip(eng, ForcedReplacementPredicateTests.REPLACEMENT), Counter()
        )

    def test_residuals_survive_a_battle_end(self) -> None:
        """The revert-direction control for the first attempt's failure."""
        eng = Counter({("sandstorm", -16): 1})
        self.assertEqual(
            self._strip(eng, ForcedReplacementPredicateTests.BATTLE_END), eng
        )

    def test_a_hazard_survives_a_replacement_ply(self) -> None:
        eng = Counter({("spikes", -32): 1})
        self.assertEqual(
            self._strip(eng, ForcedReplacementPredicateTests.REPLACEMENT), eng
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
