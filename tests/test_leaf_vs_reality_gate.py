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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from leaf_vs_reality import (  # noqa: E402
    V4_LIVE_TENDENCY_COLUMNS,
    V4_MATCHUP_PAIR_COLUMNS,
    V4_ROOT_FROZEN_PACK_COLUMNS,
    classify,
    gate_exit_code,
)


class GateExitCodeTest(unittest.TestCase):
    def test_a_clean_run_exits_zero(self) -> None:
        self.assertEqual(gate_exit_code(0, 0), 0)

    def test_matchup_excess_ALONE_fails_the_run(self) -> None:
        """The arm that cannot be observed end-to-end. Measured 470 on the frozen build."""
        self.assertEqual(gate_exit_code(0, 1), 1)
        self.assertEqual(gate_exit_code(0, 470), 1)

    def test_state_defects_alone_still_fail_the_run(self) -> None:
        """Guard the narrowness: adding the second arm must not have weakened the first."""
        self.assertEqual(gate_exit_code(124, 0), 1)

    def test_both_arms_fail(self) -> None:
        self.assertEqual(gate_exit_code(124, 470), 1)


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

    def test_turns_active_total_is_NOT_a_sibling(self) -> None:
        """It is the third member of the same triple but counts turns, not stay-or-switch events,
        and diverges on 91 of the 12-game corpus's boundaries -- letting it excuse a matchup
        divergence would loosen the subset from 4 boundaries to 95."""
        self.assertNotIn("NUMERIC_MON_TURNS_ACTIVE_TOTAL", V4_LIVE_TENDENCY_COLUMNS)

    def test_the_pair_left_the_frozen_pack(self) -> None:
        self.assertFalse(V4_MATCHUP_PAIR_COLUMNS & V4_ROOT_FROZEN_PACK_COLUMNS)

    def test_the_two_sets_are_disjoint(self) -> None:
        """The recording loop uses if/elif per family; disjointness is what makes that safe."""
        self.assertFalse(V4_MATCHUP_PAIR_COLUMNS & V4_LIVE_TENDENCY_COLUMNS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
