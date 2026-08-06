"""The differential's coverage denominator must stay recoverable from its report.

`boundaries_full_round` and `skip:single_seat_boundary` are DISJOINT: the counters
increment in the `if set(requested) == {"p1","p2"}` and `else` branches of the
boundary loop respectively, so a single-seat ply is never counted as a full round.
That makes `measured_fraction_of_full_rounds` a fraction of full rounds only — it
silently excludes ~10 % of boundaries.

Measured on the C131 artifacts: dev 1,742 of 17,710 boundaries (9.8 %) and holdout
1,813 of 17,968 (10.1 %) are single-seat and never compared, so the differential
measures ~87 % of boundaries rather than the ~96.6 % its own metric reports. The
deferred-residual population lives almost entirely in that gap: every
post-move-faint replacement ply is single-seat EXCEPT when both actives faint in the
same ply (Explosion, Selfdestruct, Destiny Bond, recoil KO), where both sides get
`forceSwitch`, the boundary is full-round, and it IS measured.

These pins keep the bound COMPUTABLE. They deliberately do not pin the fraction
itself: that drifts with the pool and the seed window, and a gate on it would fail
for the wrong reason. What must not happen is a report from which the denominator
cannot be recovered at all, because then a residue figure reads as full coverage.

See `reports/c132_single_seat_coverage_bound.md`.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

ARTIFACTS = REPO / "reports" / "artifacts"

# The two windows C131 measured. Both are committed, so this pins against real
# reports rather than a synthetic dict that could drift from the writer.
_REPORTS = (
    "c131_leechseed_main_dev_sweep.json",
    "c131_leechseed_main_holdout_sweep.json",
    "c131_leechseed_fix_dev_sweep.json",
    "c131_leechseed_fix_holdout_sweep.json",
)


class SingleSeatCoverageBoundTests(unittest.TestCase):
    def _load(self, name: str) -> dict:
        path = ARTIFACTS / name
        self.assertTrue(path.is_file(), f"missing committed artifact {name}")
        return json.loads(path.read_text())

    def test_both_counters_are_present_so_the_denominator_is_recoverable(self) -> None:
        for name in _REPORTS:
            report = self._load(name)
            counters = report.get("counters")
            self.assertIsInstance(counters, dict, f"{name}: no counters block")
            self.assertIn(
                "boundaries_full_round", counters,
                f"{name}: without this the coverage denominator cannot be recovered",
            )
            self.assertIn(
                "skip:single_seat_boundary", counters,
                f"{name}: without this the ~10% unmeasured population is invisible",
            )

    def test_the_two_counters_are_disjoint_and_sum_to_all_boundaries(self) -> None:
        # Disjointness is the whole reason `measured_fraction_of_full_rounds` is not
        # coverage. If a future edit ever counts single-seat plies as full rounds,
        # the metric silently becomes coverage and this pin should be revisited
        # deliberately rather than discovered by someone reading a residue figure.
        for name in _REPORTS:
            report = self._load(name)
            counters = report["counters"]
            full = counters["boundaries_full_round"]
            single = counters["skip:single_seat_boundary"]
            measured = report["boundaries_measured"]

            self.assertGreater(single, 0, f"{name}: expected some single-seat plies")
            self.assertGreater(full, 0, f"{name}: expected some full rounds")
            # Measured boundaries are a subset of full rounds, never of the total.
            self.assertLessEqual(
                measured, full,
                f"{name}: measured ({measured}) exceeds full rounds ({full}), so the "
                "two counters are no longer disjoint",
            )
            # THE RECONCILIATION, and it is what actually catches the regression
            # this pin claims to catch. Folding single-seat plies into
            # `boundaries_full_round` -- i.e. hoisting the increment above the
            # `if` -- escaped every other assertion here, because they all get
            # WEAKER as `full` grows. A review built that mutation and it passed:
            # my own red-run had only "caught" it because hand-editing the JSON
            # left `measured_fraction_of_full_rounds` stale at its old value,
            # which the live writer would never do.
            #
            # `full_round` must equal measured plus the exits taken INSIDE the
            # full-round path. That catches the fold-and-recompute mutation.
            #
            # It is NOT a complete guard, and the comment here used to claim it was
            # ("breaks the moment single-seat plies are counted as full rounds"). A
            # review constructed the counterexample: fold single-seat into
            # `full_round` AND route those plies to an in-path exit such as
            # `skip:no_action_candidates`, and the identity still holds while the
            # denominator becomes unrecoverable again. `exits` is a hardcoded prefix
            # allowlist, so a renamed or newly-added exit counter is in fact the
            # likeliest way for this assertion to trip.
            exits = sum(
                v
                for k, v in counters.items()
                if k.startswith(
                    (
                        "skip:world_unsupported",
                        "skip:unmappable_choice",
                        "skip:no_materialization",
                        "skip:no_action_candidates",
                        "skip:world_error",
                        "limit:",
                    )
                )
            ) + counters.get("world_prestate_mismatch", 0)
            self.assertEqual(
                measured + exits, full,
                f"{name}: the full-round path no longer reconciles "
                f"({measured} + {exits} != {full}); single-seat plies may now be "
                "counted as full rounds",
            )

            # And the reported fraction really is over full rounds, not over all
            # boundaries -- which is the fact the report exists to state.
            #
            # `assertIn` rather than a truthiness guard: the writer always emits
            # this key, and a bare `if fraction is not None` silently dropped both
            # assertions below when a review renamed it.
            self.assertIn("measured_fraction_of_full_rounds", report, f"{name}")
            fraction = report["measured_fraction_of_full_rounds"]
            self.assertAlmostEqual(
                fraction, measured / full, places=3,
                msg=(
                    f"{name}: measured_fraction_of_full_rounds does not equal "
                    "measured/full_round, so its denominator has changed meaning"
                ),
            )
            self.assertGreater(
                fraction, measured / (full + single),
                f"{name}: the reported fraction must be strictly larger than "
                "coverage over ALL boundaries, or single-seat plies are being "
                "counted somewhere they were not before",
            )

    def test_the_unmeasured_population_is_material_not_a_rounding_error(self) -> None:
        # Anti-complacency. If this ever drops near zero the bound stops mattering
        # and C132 can be retired -- but that should be an observation, not a
        # silent change. A loose lower bound catches "someone made single-seat plies
        # comparable" without pinning the exact fraction.
        for name in _REPORTS:
            report = self._load(name)
            counters = report["counters"]
            single = counters["skip:single_seat_boundary"]
            total = counters["boundaries_full_round"] + single
            share = single / total
            self.assertGreater(
                share, 0.01,
                f"{name}: single-seat share fell to {share:.4f}; if that is real, "
                "C132's bound needs revisiting rather than quietly holding",
            )
            self.assertLess(
                share, 0.5,
                f"{name}: single-seat share rose to {share:.4f}, which would mean "
                "most boundaries are uncomparable and the instrument needs review",
            )


if __name__ == "__main__":
    unittest.main()
