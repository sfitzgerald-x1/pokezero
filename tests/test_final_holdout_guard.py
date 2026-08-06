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
        # End to end through main().
        #
        # THIS PIN USED TO PASS FOR A GUARD PLACED AFTER THE SWEEP. A review moved
        # the guard to just before the report write, stubbed the simulator, and all
        # seven pins stayed green while the run measured all 60 reserved seeds. The
        # only reason it looked caught locally was an accident: the engine build was
        # stale, so `assert_fresh` raised SystemExit before the sweep. That means
        # the pin discriminated only on a machine that CANNOT sweep, and went blind
        # exactly where a sweep is possible -- which is CI.
        #
        # So: `--skip-build-check` and a nonexistent `--showdown-root`, to take
        # build freshness and dex loading out of the verdict entirely. If the guard
        # is anywhere after those, this fails with a crash instead of exit 2.
        import tempfile

        from engine_transition_differential import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "must-not-exist.json"
            ck = Path(tmp) / "must-not-exist.jsonl"
            code = main(
                [
                    "--games", "60",
                    "--seed-start", str(FINAL_HOLDOUT_SEED_FLOOR),
                    "--matcher", "strict",
                    "--skip-build-check",
                    "--showdown-root", "/nonexistent/holdout/guard/probe",
                    "--json", str(out),
                    "--checkpoint", str(ck),
                ]
            )
            self.assertEqual(code, 2, "the guard must fail with a nonzero exit code")
            self.assertFalse(
                out.exists(),
                "a refused run must not leave a report behind, or the seeds look measured",
            )
            self.assertFalse(ck.exists(), "a refused run must not leave a checkpoint either")

    def test_the_guard_fires_before_a_single_game_is_played(self) -> None:
        # The ordering property, pinned DIRECTLY.
        #
        # This needs a REAL Showdown root, and that is the whole subtlety. The pin
        # above uses a nonexistent root so build freshness and dex loading cannot
        # decide its verdict -- but that also means removing the guard makes it die
        # in `load_showdown_dex` with a FileNotFoundError, which masks the ordering
        # signal. It detects "guard absent" for the wrong reason.
        #
        # So here the dex must actually load, `run_game` must actually be
        # reachable, and only the guard may stand between the two. With the guard
        # present: exit 2, `run_game` never called. With the guard moved after the
        # sweep: `run_game` fires and `played` is non-empty, which is an ordering
        # failure and reads as one.
        import tempfile

        import engine_transition_differential as etd

        root = Path(etd.DEFAULT_SHOWDOWN_ROOT)
        if not (root / "dist" / "sim" / "index.js").is_file():
            # Deliberately NOT a silent pass. A skip here means this pin is not
            # gating, so it must be loud and it must name the reason.
            self.skipTest(
                f"built Showdown data absent under {root}; the ordering pin cannot "
                "distinguish guard-before-sweep from a dex load failure without it"
            )

        played: list[object] = []

        def _explode(*args: object, **kwargs: object) -> None:
            played.append(args)
            raise AssertionError("run_game was called on reserved seeds")

        original = etd.run_game
        etd.run_game = _explode  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = etd.main(
                    [
                        "--games", "60",
                        "--seed-start", str(FINAL_HOLDOUT_SEED_FLOOR),
                        "--matcher", "strict",
                        "--skip-build-check",
                        "--json", str(Path(tmp) / "out.json"),
                    ]
                )
        finally:
            etd.run_game = original  # type: ignore[assignment]

        self.assertEqual(played, [], "the sweep ran before the guard refused it")
        self.assertEqual(code, 2)

    def test_aggregating_reserved_seeds_is_refused(self) -> None:
        # NB-3: the argparse check bounds only what this invocation would EXECUTE.
        # In --merge-from mode --seed-start/--games are untouched defaults, so a
        # checkpoint carrying reserved seeds used to merge into an
        # `acceptance_eligible: true` report with no warning -- precisely the
        # "seeds look measured" artifact the pin above forbids.
        from engine_transition_differential import _reject_reserved_seeds_in_records

        clean = [{"seed": 19_000_000}, {"seed": 19_100_199}]
        self.assertIsNone(_reject_reserved_seeds_in_records(clean, False))

        tainted = clean + [{"seed": FINAL_HOLDOUT_SEED_FLOOR + 7}]
        message = _reject_reserved_seeds_in_records(tainted, False)
        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn(str(FINAL_HOLDOUT_SEED_FLOOR + 7), message)
        # The opt-in still works, and malformed seeds must not crash the check.
        self.assertIsNone(_reject_reserved_seeds_in_records(tainted, True))
        self.assertIsNone(
            _reject_reserved_seeds_in_records([{"seed": "not-a-number"}, {}], False)
        )

    def test_the_opt_in_attribute_is_derived_from_the_flag(self) -> None:
        # NB-4: the attribute name used to be a hand-written string duplicating the
        # flag. A review renamed the flag and all seven pins stayed green while the
        # escape hatch became unreachable, because `getattr(..., False)` read a typo
        # as "not opted in". It failed closed, which is the safe direction, but
        # silently.
        derived = FINAL_HOLDOUT_OPT_IN.removeprefix("--").replace("-", "_")
        self.assertEqual(derived, "final_holdout_i_mean_it")

        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(FINAL_HOLDOUT_OPT_IN, action="store_true")
        self.assertTrue(getattr(parser.parse_args([FINAL_HOLDOUT_OPT_IN]), derived))
        self.assertFalse(getattr(parser.parse_args([]), derived))


if __name__ == "__main__":
    unittest.main()
