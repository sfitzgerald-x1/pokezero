"""The denominator rule: a run that measured nothing is not a pass.

Every differential harness in this repo has at some point reported a pass it had no ability
to withhold — `leaf_vs_reality` printing `divergent boundaries: 0` while skipping 100% of
them, `leaf_root_parity` and `prior_mapping_assert` exiting on predicates that are vacuously
true over zero rows, and `fidelity_gate_events` ending in an unconditional `return 0`. This
pins the shared helper that makes those impossible.

The acceptance test for the whole rule is `test_a_run_that_measured_nothing_exits_nonzero`
plus its adoption twin in `tests/test_denominator_adoption.py`, which drives the four real
harnesses with every boundary skipped.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from differential_denominator import (  # noqa: E402
    DenominatorReport,
    check_denominator,
    gate,
)


class ZeroDenominatorTest(unittest.TestCase):
    """Rule 2 — the half no caller can fake."""

    def test_a_run_that_measured_nothing_exits_nonzero(self) -> None:
        """THE acceptance pin. A harness that skipped everything must not report a pass."""
        report = check_denominator("corpus", measured=0, matched=0, diverged=0)
        self.assertFalse(report.ok)
        self.assertTrue(report.inert)
        self.assertEqual(gate([report]), 1)

    def test_the_old_shape_would_have_passed(self) -> None:
        """Non-vacuity for the test above: the predicate every harness used really is
        vacuously true here, so the pin is catching something that used to slip through."""
        matched, diverged = 0, 0
        self.assertTrue(diverged == 0, "the old `all(divergent == 0)` gate")
        self.assertEqual(matched + diverged, 0)
        self.assertEqual(gate([check_denominator("c", measured=0, matched=0, diverged=0)]), 1)

    def test_no_reports_at_all_is_also_a_failure(self) -> None:
        """A harness that produced no corpus reports measured nothing either."""
        self.assertEqual(gate([]), 1)

    def test_the_failure_names_the_corpus_and_the_skips(self) -> None:
        report = check_denominator("gv2", measured=0, matched=0, diverged=0, contained=1008, skipped=1008)
        message = report.failures[0]
        self.assertIn("gv2", message)
        self.assertIn("1008", message)
        self.assertIn("not a pass", message)


class PartitionTest(unittest.TestCase):
    """Rule 3 — every attempted boundary is classified exactly once."""

    def test_an_unclassified_boundary_fails(self) -> None:
        report = check_denominator("c", measured=10, matched=4, diverged=5)
        self.assertFalse(report.ok)
        self.assertIn("never classified", report.failures[0])

    def test_a_double_classified_boundary_fails(self) -> None:
        report = check_denominator("c", measured=10, matched=6, diverged=5)
        self.assertFalse(report.ok)
        self.assertIn("classified twice", report.failures[0])

    def test_a_clean_partition_passes(self) -> None:
        report = check_denominator("c", measured=10, matched=4, diverged=6)
        self.assertTrue(report.ok)
        self.assertFalse(report.inert)
        self.assertEqual(gate([report]), 0)

    def test_both_rules_can_fire_together(self) -> None:
        """Zero measured with a nonzero classification is incoherent twice over."""
        report = check_denominator("c", measured=0, matched=1, diverged=0)
        self.assertEqual(len(report.failures), 2)


class SpanningRuleTest(unittest.TestCase):
    """Rule 4 — the one that is not structurally forced.

    Rule 3 sits entirely on one side of the skip decision, so an increment adjacent to its
    classification can never violate it (true at all four adoption sites). Rule 4 spans that
    decision, and it catches both bugs this PR's first review round found.
    """

    def test_it_catches_a_counter_placed_above_the_skip_decision(self) -> None:
        """Round-1 `fidelity_gate_events`: `attempted` incremented before `drive_boundary`,
        so it equalled the contained count while 315 boundaries were skipped."""
        report = check_denominator(
            "gv4", measured=1271, matched=459, diverged=812, contained=1271, skipped=315
        )
        self.assertFalse(report.ok)
        self.assertIn("counted twice", " ".join(report.failures))

    def test_it_catches_a_boundary_counted_as_both_compared_and_skipped(self) -> None:
        """Round-1 `leaf_vs_reality`: the `no_golden_row` path incremented `attempted` and
        then `skip:no_golden_row`."""
        report = check_denominator(
            "gv4", measured=956, matched=39, diverged=917, contained=1271, skipped=316
        )
        self.assertFalse(report.ok)
        self.assertIn("counted twice", " ".join(report.failures))

    def test_it_catches_a_boundary_that_is_neither_compared_nor_skipped(self) -> None:
        report = check_denominator(
            "gv4", measured=956, matched=39, diverged=917, contained=1271, skipped=314
        )
        self.assertFalse(report.ok)
        self.assertIn("neither compared nor skipped", " ".join(report.failures))

    def test_the_real_shape_passes(self) -> None:
        self.assertTrue(
            check_denominator(
                "gv4", measured=956, matched=39, diverged=917, contained=1271, skipped=315
            ).ok
        )

    def test_it_is_skipped_when_either_figure_is_absent(self) -> None:
        """Callers that cannot supply contained/skipped must keep working."""
        for kwargs in ({"contained": 1271}, {"skipped": 315}, {}):
            with self.subTest(**kwargs):
                self.assertTrue(
                    check_denominator("c", measured=956, matched=39, diverged=917, **kwargs).ok
                )


class RenderTest(unittest.TestCase):
    def test_the_denominator_line_carries_the_numbers_a_reader_needs(self) -> None:
        line = check_denominator(
            "gv4", measured=956, matched=39, diverged=917, contained=1271, skipped=315
        ).render()
        for token in ("956", "1271", "315", "39", "917", "gv4"):
            self.assertIn(token, line)

    def test_gate_prints_every_report_before_failing(self) -> None:
        """A failing corpus must not suppress the others' denominators."""
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = gate(
                [
                    check_denominator("good", measured=4, matched=4, diverged=0),
                    check_denominator("empty", measured=0, matched=0, diverged=0),
                ]
            )
        printed = buffer.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("good", printed)
        self.assertIn("empty", printed)


class ContractTest(unittest.TestCase):
    """What the helper deliberately cannot enforce, recorded so nobody assumes it does."""

    def test_a_caller_that_derives_measured_from_the_sum_is_undetectable(self) -> None:
        """`measured = matched + diverged` makes rule 3 unfalsifiable, and the helper CANNOT
        see it — 5 == 2 + 3 looks identical however `measured` was obtained.

        This is the tautology `leaf_vs_reality` shipped: `compared` was assigned
        `exact + divergent` and the identity was then cited as evidence. The defence is a
        contract on the caller, stated in the module docstring and at each adoption site, and
        this test exists so the limitation is written down rather than assumed away.
        """
        matched, diverged = 2, 3
        derived = check_denominator("derived", measured=matched + diverged, matched=matched, diverged=diverged)
        independent = check_denominator("independent", measured=5, matched=matched, diverged=diverged)
        self.assertTrue(derived.ok)
        self.assertTrue(independent.ok)
        self.assertEqual(derived.failures, independent.failures)

    def test_a_caller_that_derives_contained_makes_rule_4_tautological(self) -> None:
        """The same hole, one rule over -- and rule 4 is the load-bearing one.

        Review demonstrated this live: replacing `contained=report["boundaries"]` with
        `contained = attempted + sum(skips)` in leaf_vs_reality leaves rule 4 permanently
        satisfied and the whole suite green. The helper cannot see it, for the same reason it
        cannot see a derived `measured`: 5 == 2 + 3 looks identical however it was obtained.

        So `contained` must be the corpus's OWN count of what it holds, read from the corpus
        rather than reconstructed from the harness's own bookkeeping -- otherwise rule 4 checks
        the harness against itself. All four adoption sites read it from the report's corpus
        figure; nothing here enforces that, and this test exists so the gap is written down
        rather than assumed shut.
        """
        derived = check_denominator(
            "derived", measured=956, matched=39, diverged=917, contained=956 + 315, skipped=315
        )
        independent = check_denominator(
            "independent", measured=956, matched=39, diverged=917, contained=1271, skipped=315
        )
        self.assertTrue(derived.ok)
        self.assertEqual(derived.failures, independent.failures)

    def test_but_the_zero_rule_still_binds_a_deriving_caller(self) -> None:
        """The half that survives a lazy caller: if it measured nothing, `matched + diverged`
        is 0 too, so the derived `measured` is 0 and rule 2 fires anyway."""
        self.assertEqual(gate([check_denominator("d", measured=0 + 0, matched=0, diverged=0)]), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
