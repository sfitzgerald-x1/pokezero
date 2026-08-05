"""A1: a forced-replacement ply ran no residual phase (C116 Phase 3 item 9).

Showdown gives a Pokemon arriving on a forced replacement no residual tick --
the replacement completes the PREVIOUS turn rather than starting a new one. The
engine, asked for a full turn, faithfully runs one and over-emits. This is a
HARNESS change, measured and landed separately from any fidelity change.

The pins below exercise the SHIPPED predicate and the SHIPPED source set, not a
local reimplementation of either. An earlier draft of this file reimplemented the
filter inline, which would have kept passing had the real one been deleted.
"""

from __future__ import annotations

import inspect
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import engine_transition_differential as diff  # noqa: E402


def _predicate_source() -> str:
    """The shipped predicate, read off the shipped function."""
    return inspect.getsource(diff.evaluate_boundary_strict)


class PredicateShapeTests(unittest.TestCase):
    """The four clauses are each load-bearing; none may be dropped."""

    def test_the_predicate_exists_and_is_named(self) -> None:
        self.assertIn("boundary_is_forced_replacement", _predicate_source())

    def test_it_requires_the_absence_of_win(self) -> None:
        """This clause is why the first attempt failed and it must never go.

        `|upkeep|` is also absent whenever the battle ENDS during residuals --
        Showdown emits the tick, then `|faint|`, then `|win|`. Without the
        `|win|` clause the filter stripped the engine's residuals there and
        manufactured 44 divergences on a 400-game measurement, including
        reopening 19100002/53, the battle-end sandstorm row #1092 had fixed.
        """
        source = _predicate_source()
        start = source.index("boundary_is_forced_replacement = (")
        # Slice to the line that closes the assignment: a lone `    )`.
        end = source.index("\n    )\n", start)
        window = source[start:end]
        for clause in ('"|upkeep"', '"|win"', '"|move|"', '"|switch|"'):
            self.assertIn(clause, window, f"predicate lost its {clause} clause")
        # And the clauses must be conjoined, not alternatives.
        self.assertNotIn(" or ", window)
        self.assertEqual(window.count(" and "), 3, "expected four ANDed clauses")


class ResidualSourceSetTests(unittest.TestCase):
    def test_it_covers_every_source_the_a1_rows_carry(self) -> None:
        # 19000020/50 itemleftovers, 19000059/27 psn,
        # 19100181/45 itemleftovers + psn + sandstorm.
        for source in ("itemleftovers", "psn", "sandstorm"):
            self.assertIn(source, diff._RESIDUAL_PHASE_SOURCES)

    def test_hazards_are_excluded_so_a_different_cause_stays_divergent(self) -> None:
        """19100180/24 is a hazard mis-attribution, not a residual.

        Medicham switches into Spikes and faints; the engine attributes the
        hazard to a side whose active never changed. That boundary also has no
        `|upkeep|`, so only the SOURCE scoping keeps it divergent.
        """
        for hazard in ("spikes", "stealthrock"):
            self.assertNotIn(hazard, diff._RESIDUAL_PHASE_SOURCES)

    def test_it_is_not_the_majority_override_set(self) -> None:
        """`_ADJUDICABLE_RESIDUALS` serves a different rule and under-covers."""
        self.assertNotEqual(diff._RESIDUAL_PHASE_SOURCES, diff._ADJUDICABLE_RESIDUALS)
        for source in ("leechseed", "movewish", "hail"):
            self.assertIn(source, diff._RESIDUAL_PHASE_SOURCES)
            self.assertNotIn(source, diff._ADJUDICABLE_RESIDUALS)


class FilterIsWiredIntoTheComparisonTests(unittest.TestCase):
    """The filter must sit on the exact-component comparison, gated correctly."""

    def test_the_filter_guards_the_exact_comparison(self) -> None:
        source = _predicate_source()
        self.assertIn("if boundary_is_forced_replacement:", source)
        guard = source.index("if boundary_is_forced_replacement:")
        compare = source.index("if eng_exact != obs_exact_branch:")
        self.assertLess(guard, compare, "the filter must run BEFORE the comparison")
        between = source[guard:compare]
        self.assertIn("_RESIDUAL_PHASE_SOURCES", between)
        self.assertIn("eng_exact", between)

    def test_only_the_engine_side_is_filtered(self) -> None:
        """Filtering the observed side too would hide real under-emissions."""
        source = _predicate_source()
        guard = source.index("if boundary_is_forced_replacement:")
        between = source[guard : source.index("if eng_exact != obs_exact_branch:")]
        self.assertNotIn("obs_exact_branch =", between)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
