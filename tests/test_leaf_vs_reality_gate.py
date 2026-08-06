"""The leaf-vs-reality exit gate, including the matchup arm that cannot be observed end-to-end.

`scripts/leaf_vs_reality.py` grew a second gated quantity: `matchup_excess`, the boundaries where
the fold-driven matchup pair diverges from reality but no already-live tendency counter does. The
pair itself is allowed to diverge -- a leaf fold advanced over synthesized lines can legitimately
count an event reality did not, which is the documented `fold` class -- so what is gated is the
RELATIONSHIP that made surfacing the counters correct rather than a raw count.

Why this file exists: on any corpus that exercises those columns, the state class is already nonzero
(124 boundaries on the 12-game v4 corpus), so `main()`'s exit code is 1 regardless and the matchup
arm's contribution is invisible. Independent review flagged that the gate had been verified only by
reading its printed count. `gate_exit_code` is the entire decision, so it can be pinned directly.
"""
import sys
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# `leaf_vs_reality` imports numpy and pokezero_search at module scope, so a bare interpreter turns
# this file into a pytest COLLECTION ERROR while every sibling in this area degrades to a skip.
# Sharp irony worth avoiding: `gate_exit_code` was extracted for testability and its test could not
# be collected without a compiled wheel it has nothing to do with.
pytest.importorskip("numpy", reason="leaf_vs_reality imports numpy at module scope")
pytest.importorskip(
    "pokezero_search", reason="leaf_vs_reality imports the native crate at module scope"
)

from leaf_vs_reality import (  # noqa: E402
    MATCHUP_EXCESS_ALLOWANCE,
    matchup_excess_allowance,
    V4_LIVE_TENDENCY_COLUMNS,
    V4_MATCHUP_PAIR_COLUMNS,
    V4_ROOT_FROZEN_PACK_COLUMNS,
    classify,
    gate_exit_code,
)


class GateExitCodeTest(unittest.TestCase):
    def test_a_clean_run_exits_zero(self) -> None:
        self.assertEqual(gate_exit_code(0, 0), 0)

    def test_a_regression_sized_excess_fails_the_run(self) -> None:
        """The arm that cannot be observed end-to-end. Measured 425 on the frozen build."""
        self.assertEqual(gate_exit_code(0, 425), 1)
        self.assertEqual(gate_exit_code(0, MATCHUP_EXCESS_ALLOWANCE + 1), 1)

    def test_the_documented_false_positive_class_does_NOT_fail_the_run(self) -> None:
        """The allowance exists because the harness documents facing-ordering divergences as
        legitimate. Gating those at zero while the docs call them expected is how a gate ends up
        disabled -- independent review's point, and the reason this is a magnitude test."""
        self.assertEqual(gate_exit_code(0, 1), 0)
        self.assertEqual(gate_exit_code(0, MATCHUP_EXCESS_ALLOWANCE), 0)

    def test_the_ceiling_does_not_bind_on_any_corpus_in_use(self) -> None:
        """Recorded rather than hidden: the rate and the ceiling collapse to one constant below
        1280 compared boundaries, which is every corpus this runs on today. Citing the ceiling as
        the reason the rate is safe was the circular framing review caught."""
        self.assertEqual(matchup_excess_allowance(1279), 15)
        self.assertEqual(matchup_excess_allowance(1280), MATCHUP_EXCESS_ALLOWANCE)

    def test_the_allowance_ceiling_sits_below_the_measured_regression(self) -> None:
        """One bound, not two. The upper bound is measured -- a desurfacing regression puts excess
        at 425 -- but the lower bound is UNCONSTRAINED, because measured noise is 0 and so every
        positive allowance is trivially "above the noise". 16 is chosen, not derived."""
        self.assertGreater(MATCHUP_EXCESS_ALLOWANCE, 0)
        self.assertLess(MATCHUP_EXCESS_ALLOWANCE, 425)


class AllowanceScalesWithCorpusSizeTest(unittest.TestCase):
    """A FIXED allowance makes the arm structurally unable to fire on a small corpus.

    `excess <= matchup_divergent_boundaries <= boundaries`, so a flat 16 means any corpus with
    fewer than 16 boundaries returns a pass for a totally desurfaced encoder. Taking `max` over
    corpora stops sum-dilution but not shard-dilution: split one corpus into 40 pieces and a flat
    allowance is dead on every piece. Independent review found this; it is the third form of silent
    inertness in this harness, after the schema-guard skip and the un-gated prefix class.
    """

    def test_a_tiny_corpus_can_still_fail(self) -> None:
        """The committed sample is 3 boundaries. A total regression there must not pass."""
        allowance = matchup_excess_allowance(3)
        self.assertEqual(allowance, 1)
        self.assertEqual(gate_exit_code(0, 3, allowance), 1)
        self.assertEqual(gate_exit_code(0, 2, allowance), 1)

    def test_it_never_reaches_zero(self) -> None:
        """Zero would re-create the incoherence the allowance exists to remove."""
        for boundaries in (0, 1, 2, 79, 80):
            with self.subTest(boundaries=boundaries):
                self.assertGreaterEqual(matchup_excess_allowance(boundaries), 1)

    def test_it_is_capped_at_the_ceiling(self) -> None:
        self.assertEqual(matchup_excess_allowance(10_000_000), MATCHUP_EXCESS_ALLOWANCE)

    def test_the_measured_corpus_lands_under_the_ceiling_and_over_the_noise(self) -> None:
        """952 COMPARED boundaries -> 11, against a measured excess of 0 surfaced and 425 frozen.

        Compared, not contained: the 12-game corpus holds 1271 boundaries but 319 skip.
        """
        allowance = matchup_excess_allowance(952)
        self.assertEqual(allowance, 11)
        self.assertEqual(gate_exit_code(0, 0, allowance), 0)
        self.assertEqual(gate_exit_code(0, 425, allowance), 1)

    def test_sharding_the_corpus_cannot_kill_the_gate(self) -> None:
        """The dilution attack a flat allowance loses to: 1271 boundaries split 40 ways is ~31
        each, and a proportional share of a 425-excess regression is ~10 -- which must still fire
        against the scaled allowance of 1, and would NOT against a flat 16."""
        allowance = matchup_excess_allowance(1271 // 40)
        self.assertEqual(gate_exit_code(0, 425 // 40, allowance), 1)
        self.assertEqual(gate_exit_code(0, 425 // 40, MATCHUP_EXCESS_ALLOWANCE), 0)

    def test_state_defects_alone_still_fail_the_run(self) -> None:
        """Guard the narrowness: `state`/`turn` keep a ZERO threshold. The allowance is the matchup
        arm's alone, and adding it must not have loosened the original gate."""
        self.assertEqual(gate_exit_code(124, 0), 1)
        self.assertEqual(gate_exit_code(1, 0), 1)

    def test_both_arms_fail(self) -> None:
        self.assertEqual(gate_exit_code(124, 425), 1)


class MatchupClassificationTest(unittest.TestCase):
    """The pair must not be swept into `fold` by the NUMERIC_MON_* prefix coincidence.

    That is what made the first revision's gate vacuous: the columns left
    V4_ROOT_FROZEN_PACK_COLUMNS and landed in an accepted, un-gated class, so a full regression to
    the frozen behaviour would have exited 0.
    """

    def _classify(self, column: str) -> str:
        return classify("numeric_features", "opponent_team", column, False, set())

    def test_the_pair_gets_its_own_label(self) -> None:
        for column in sorted(V4_MATCHUP_PAIR_COLUMNS):
            with self.subTest(column=column):
                self.assertEqual(self._classify(column), "matchup_fold")

    def test_the_pair_is_named_so_the_prefix_rule_WOULD_have_swept_it(self) -> None:
        """Non-vacuity: if these columns stopped matching the prefix, the test above would pass
        for a trivial reason and the ordering it protects would no longer matter."""
        for column in sorted(V4_MATCHUP_PAIR_COLUMNS):
            with self.subTest(column=column):
                self.assertTrue(column.startswith("NUMERIC_MON_"))

    def test_the_live_tendency_siblings_still_classify_as_fold(self) -> None:
        for column in sorted(V4_LIVE_TENDENCY_COLUMNS):
            with self.subTest(column=column):
                self.assertEqual(self._classify(column), "fold")

    def test_the_whole_tendency_triple_is_a_sibling(self) -> None:
        """Including TURNS_ACTIVE_TOTAL, which an earlier revision excluded on false arithmetic.

        The claim was that including it would "let most of a full regression through". Measured:
        frozen excess is 470 with two siblings and 425 with three, against a surfaced excess of 0
        either way -- so the exclusion bought no detection while costing false-alarm headroom on
        the sibling most correlated with the known false-positive class (its 91/1271 divergences
        are evidence that leaf occupant attribution drifts, and occupant attribution is where the
        matchup cell's `facing` key comes from).
        """
        self.assertIn("NUMERIC_MON_TURNS_ACTIVE_TOTAL", V4_LIVE_TENDENCY_COLUMNS)

    def test_the_pair_left_the_frozen_pack(self) -> None:
        self.assertFalse(V4_MATCHUP_PAIR_COLUMNS & V4_ROOT_FROZEN_PACK_COLUMNS)

    def test_the_two_sets_are_disjoint(self) -> None:
        """The recording loop uses if/elif per family; disjointness is what makes that safe."""
        self.assertFalse(V4_MATCHUP_PAIR_COLUMNS & V4_LIVE_TENDENCY_COLUMNS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
