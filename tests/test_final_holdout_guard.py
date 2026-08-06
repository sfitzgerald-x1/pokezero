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


def _reserved_record(seed: int) -> dict[str, object]:
    """A schema-valid checkpoint record, so `load_checkpoint` accepts it.

    The point of these two pins is to reach the RECORD-level guard through
    `main()`, so the record has to survive schema validation first -- a malformed
    one is rejected earlier for the wrong reason and would pin nothing.
    """

    from engine_transition_differential import CHECKPOINT_SCHEMA

    return {
        "schema": CHECKPOINT_SCHEMA,
        "build_check": "gated",
        "provenance": {"engine_fingerprint": None, "image_commit": None, "source_commit": None},
        "seed": int(seed),
        "seconds": 0.0,
        "counters": {},
        "repros": [],
    }


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

    def test_the_message_names_the_registered_block_only_where_that_is_true(self) -> None:
        # NB-3 from round 2: swapping the two rationale branches was undetected by
        # every pin. The distinction is not cosmetic -- "reserved for exactly ONE
        # measurement, ever" is FALSE for already-consumed historical bands above the
        # floor (c73 swept 19,500,000), and a guard that tells a contributor a
        # falsehood about their own seeds teaches them to distrust it.
        registered = _reject_unguarded_final_holdout(FINAL_HOLDOUT_SEED_FLOOR, 60, False)
        assert registered is not None
        self.assertIn("registered final-holdout block", registered)
        self.assertIn("exactly ONE measurement", registered)

        above = _reject_unguarded_final_holdout(19_500_000, 10, False)
        assert above is not None
        self.assertIn("reserved by default", above)
        self.assertNotIn(
            "exactly ONE measurement", above,
            "a consumed band above the floor must not be described as the one-shot block",
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
        # THE ORDERING PROPERTY, and this pin must hold WITHOUT a Showdown checkout.
        #
        # My first version required the real Showdown root and called `skipTest`
        # otherwise. CI has no built Showdown -- no workflow installs node or builds
        # it, and two other steps in the same job say so outright -- so the pin
        # skipped, the workflow's own no-skip assertion fired, and I turned the
        # required check RED on every PR that trips the filter. The only route back
        # to green would have been to weaken the step: the exact failure mode this
        # whole change exists to prevent, one level up. Round 2's review caught it
        # from the live run.
        #
        # So the collaborators between the guard and the simulator are stubbed
        # instead. The pin now discriminates on a machine that CANNOT sweep, which
        # is precisely what CI is.
        import tempfile

        import engine_transition_differential as etd

        played: list[object] = []

        def _explode(*args: object, **kwargs: object) -> None:
            played.append(args)
            raise AssertionError("run_game was called on reserved seeds")

        def _unreachable(name: str):
            def _fail(*args: object, **kwargs: object) -> None:
                raise AssertionError(f"{name} ran before the guard refused the seeds")

            return _fail

        saved = {
            name: getattr(etd, name)
            for name in ("run_game", "load_showdown_dex", "LocalShowdownEnv", "EngineMctsPolicy")
        }
        etd.run_game = _explode  # type: ignore[assignment]
        etd.load_showdown_dex = _unreachable("load_showdown_dex")  # type: ignore[assignment]
        etd.LocalShowdownEnv = _unreachable("LocalShowdownEnv")  # type: ignore[assignment]
        etd.EngineMctsPolicy = _unreachable("EngineMctsPolicy")  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "must-not-exist.json"
                code = etd.main(
                    [
                        "--games", "60",
                        "--seed-start", str(FINAL_HOLDOUT_SEED_FLOOR),
                        "--matcher", "strict",
                        "--skip-build-check",
                        # A NONEXISTENT root, deliberately. This is the only place in
                        # the repo that calls `main()` at reserved seeds, and without
                        # this it targets DEFAULT_SHOWDOWN_ROOT -- a real checkout on a
                        # developer machine. A refactor that keeps these four names
                        # while routing around them (a local import, say) would bind
                        # the stubs to nothing, and combined with a guard regression
                        # the test suite itself would sweep 60 reserved seeds. With it,
                        # this pin CANNOT sweep, and it still fails on the ordering
                        # assertion under every mutation.
                        "--showdown-root", "/nonexistent/holdout/guard/ordering",
                        "--json", str(out),
                    ]
                )
                self.assertFalse(out.exists(), "a refused run must write no report")
        finally:
            for name, original in saved.items():
                setattr(etd, name, original)

        # `load_showdown_dex` is the FIRST tripwire, not `run_game`: it is called
        # ~20 lines earlier, so under every mutation I built it is what actually
        # fires ("load_showdown_dex ran before the guard refused the seeds"). An
        # earlier version of this comment, and the PR body, claimed `run_game` was
        # the failing assertion -- stale, and it made `played` read as load-bearing
        # when it is unreachable. `played` stays as a labelled backstop for a future
        # refactor that reorders those calls.
        self.assertEqual(played, [], "the sweep ran before the guard refused it")
        self.assertEqual(code, 2)

    def test_merging_a_reserved_seed_checkpoint_is_refused_through_main(self) -> None:
        # Round 2's review found that deleting the record guard from EITHER
        # aggregation site escaped all ten pins, because the only coverage drove the
        # helper directly and never `main()`. Two pins now drive `main()`.
        import json
        import tempfile

        from engine_transition_differential import main

        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "reserved.jsonl"
            ck.write_text(
                "\n".join(
                    json.dumps(_reserved_record(FINAL_HOLDOUT_SEED_FLOOR + i))
                    for i in range(3)
                )
                + "\n"
            )
            out = Path(tmp) / "must-not-exist.json"
            code = main(["--merge-from", str(ck), "--json", str(out)])
            self.assertEqual(code, 2, "merging reserved seeds must be refused")
            self.assertFalse(
                out.exists(),
                "a refused merge must not leave a report that makes the seeds look measured",
            )

    def test_resuming_a_reserved_seed_checkpoint_is_refused_through_main(self) -> None:
        import json
        import tempfile

        from engine_transition_differential import main

        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "reserved.jsonl"
            ck.write_text(json.dumps(_reserved_record(FINAL_HOLDOUT_SEED_FLOOR)) + "\n")
            out = Path(tmp) / "must-not-exist.json"
            # Clean args on purpose: seed_start is the DEV window, so only the
            # record-level guard can catch this.
            code = main(
                [
                    "--checkpoint", str(ck),
                    "--resume",
                    "--seed-start", "19000000",
                    "--games", "1",
                    "--skip-build-check",
                    "--json", str(out),
                ]
            )
            self.assertEqual(code, 2, "resuming over reserved seeds must be refused")
            self.assertFalse(out.exists(), "a refused resume must write no report")

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

    def test_the_record_guard_fails_closed_on_non_integer_seeds(self) -> None:
        # NB-2 from round 2, and it FAILED OPEN, which is the worst direction. The
        # original check was `str(r["seed"]).lstrip("-").isdigit()`, so a record
        # carrying `19200008.0` or `" 19200009"` was skipped entirely and an
        # `acceptance_eligible` report over the reserved range was written with exit
        # 0. Floats genuinely reach here -- the dedupe upstream already does
        # `int(record.get("seed", -1))`.
        #
        # I fixed the coercion and then did NOT pin it: a mutation restoring
        # `.isdigit()` escaped all twelve pins. This is that pin.
        from engine_transition_differential import _reject_reserved_seeds_in_records

        for seed in (
            FINAL_HOLDOUT_SEED_FLOOR + 8,          # int
            float(FINAL_HOLDOUT_SEED_FLOOR + 8),   # float, the reported hole
            f" {FINAL_HOLDOUT_SEED_FLOOR + 9}",    # leading whitespace
            str(FINAL_HOLDOUT_SEED_FLOOR + 10),    # plain string
        ):
            self.assertIsNotNone(
                _reject_reserved_seeds_in_records([{"seed": seed}], False),
                f"a reserved seed expressed as {seed!r} slipped through",
            )

        # Unparseable seeds must not crash the check; they are simply not evidence
        # of a reserved run.
        for junk in ("not-a-number", None, object()):
            self.assertIsNone(_reject_reserved_seeds_in_records([{"seed": junk}], False))
        self.assertIsNone(_reject_reserved_seeds_in_records([{}], False))

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
