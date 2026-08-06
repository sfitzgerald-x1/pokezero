"""The reserved final holdout must be unreachable without an explicit opt-in.

Written after I ran 60 games against it. The reservation had lived only in a plan
document and in whoever was driving the script; a convenience shell loop over three
`--seed-start` values (`for start in 19100000 19000000 19200000`) executed
`19,200,000`-`19,200,059` while probing an unrelated question. Nothing about that
probe needed the reserved range.

A window whose entire value is that it has never been executed should not be
reachable by a typo or a loop. These pins exist so the enforcement lives in the
tool.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from engine_transition_differential import (  # noqa: E402
    FINAL_HOLDOUT_OPT_IN,
    FINAL_HOLDOUT_SEED_FLOOR,
    _reject_unguarded_final_holdout,
)


class FinalHoldoutGuardTests(unittest.TestCase):
    def test_the_two_working_windows_are_never_blocked(self) -> None:
        # The dev and validation windows are swept constantly. If the guard ever
        # touches them it will be ripped out, and then it protects nothing.
        self.assertIsNone(_reject_unguarded_final_holdout(19_000_000, 200, False))
        self.assertIsNone(_reject_unguarded_final_holdout(19_100_000, 200, False))

    def test_the_incident_command_is_refused(self) -> None:
        # Verbatim the run that caused this file to exist.
        message = _reject_unguarded_final_holdout(19_200_000, 60, False)
        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("19200000..19200059", message)
        self.assertIn(FINAL_HOLDOUT_OPT_IN, message)

    def test_a_span_that_reaches_the_floor_is_refused_even_when_it_starts_below(self) -> None:
        # THE CASE A NAIVE GUARD MISSES. `seed_start >= FLOOR` would wave this
        # through: it starts 10 seeds below the floor and runs 190 seeds past it.
        message = _reject_unguarded_final_holdout(19_199_990, 200, False)
        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("19200000..19200189", message)

        # And the exact-fit boundary: the 200th seed IS the floor, so it counts.
        self.assertIsNotNone(
            _reject_unguarded_final_holdout(FINAL_HOLDOUT_SEED_FLOOR - 199, 200, False)
        )
        # One seed earlier ends at FLOOR - 1 and must be allowed.
        self.assertIsNone(
            _reject_unguarded_final_holdout(FINAL_HOLDOUT_SEED_FLOOR - 200, 200, False)
        )

    def test_a_single_game_exactly_on_the_floor_is_refused(self) -> None:
        # `--games 1` must not slip through an off-by-one in the span arithmetic.
        self.assertIsNotNone(
            _reject_unguarded_final_holdout(FINAL_HOLDOUT_SEED_FLOOR, 1, False)
        )
        self.assertIsNone(
            _reject_unguarded_final_holdout(FINAL_HOLDOUT_SEED_FLOOR - 1, 1, False)
        )

    def test_games_zero_still_counts_as_touching_the_floor(self) -> None:
        # `max(games, 1)` means a degenerate 0-game run is still judged on its
        # start seed rather than silently computing a span that ends before it
        # begins.
        self.assertIsNotNone(
            _reject_unguarded_final_holdout(FINAL_HOLDOUT_SEED_FLOOR, 0, False)
        )

    def test_the_opt_in_is_the_only_way_through(self) -> None:
        self.assertIsNone(
            _reject_unguarded_final_holdout(FINAL_HOLDOUT_SEED_FLOOR, 200, True)
        )

    def test_the_cli_refuses_with_a_nonzero_exit_and_writes_no_report(self) -> None:
        # End to end through main(), because the guard is only worth anything if it
        # fires before any measurement happens and before any JSON is emitted.
        import tempfile

        from engine_transition_differential import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "must-not-exist.json"
            code = main(
                [
                    "--games", "60",
                    "--seed-start", str(FINAL_HOLDOUT_SEED_FLOOR),
                    "--matcher", "strict",
                    "--json", str(out),
                ]
            )
            self.assertEqual(code, 2, "the guard must fail with a nonzero exit code")
            self.assertFalse(
                out.exists(),
                "a refused run must not leave a report behind, or the seeds look measured",
            )


if __name__ == "__main__":
    unittest.main()
