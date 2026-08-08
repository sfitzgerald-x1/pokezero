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
    BURNED_FINAL_HOLDOUT,
    FINAL_HOLDOUT_OPT_IN,
    FINAL_HOLDOUT_SEED_FLOOR,
    OWNER_RATIFIED,
    RATIFIED_FINAL_HOLDOUT,
    RATIFIED_SWEEP_PRECONDITION,
    _reject_burned_final_holdout,
    _reject_unguarded_final_holdout,
)

# A seed that is RESERVED but not BURNED, for the pins that need the opt-in to still
# work. `FINAL_HOLDOUT_SEED_FLOOR + 7` used to serve this purpose and no longer can:
# C151 burned `19,200,000`-`19,200,259` unconditionally, so that seed is now refused
# whatever the opt-in says. Deriving it from the ratified band rather than hardcoding
# means it follows the band if the band ever moves.
RESERVED_NOT_BURNED = RATIFIED_FINAL_HOLDOUT[0] + 7


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

    def test_the_message_names_the_right_block_in_each_of_the_three_cases(self) -> None:
        # NB-3 from round 2: swapping the rationale branches was undetected by every
        # pin. The distinction is not cosmetic -- "reserved for exactly ONE
        # measurement, ever" is FALSE for already-consumed historical bands above the
        # floor (c73 swept 19,500,000), and a guard that tells a contributor a
        # falsehood about their own seeds teaches them to distrust it.
        #
        # C151 makes it three branches, not two. `19,200,000` used to be the registered
        # block and is now BURNED, so the branch that fires there changed meaning; a pin
        # that still said "registered final-holdout block" here would be asserting the
        # old semantics against the new guard.
        burned = _reject_unguarded_final_holdout(FINAL_HOLDOUT_SEED_FLOOR, 60, False)
        assert burned is not None
        self.assertIn("BURNED", burned)
        self.assertNotIn("registered final-holdout block", burned)

        ratified = _reject_unguarded_final_holdout(RATIFIED_FINAL_HOLDOUT[0], 200, False)
        assert ratified is not None
        self.assertIn("ratified final-holdout block", ratified)
        self.assertIn("exactly ONE measurement", ratified)
        self.assertIn(OWNER_RATIFIED[1], ratified)

        above = _reject_unguarded_final_holdout(19_500_000, 10, False)
        assert above is not None
        self.assertIn("reserved by default", above)
        self.assertNotIn(
            "exactly ONE measurement", above,
            "a consumed band above the floor must not be described as the one-shot block",
        )
        self.assertNotIn("BURNED", above)

    def test_the_opt_in_opens_the_ratified_window_and_nothing_else(self) -> None:
        # The opt-in exists to permit the ONE ratified terminal measurement.
        low, high = RATIFIED_FINAL_HOLDOUT
        self.assertIsNone(_reject_unguarded_final_holdout(low, high - low + 1, True))
        # And it is still required: without it the ratified window is refused.
        self.assertIsNotNone(_reject_unguarded_final_holdout(low, 200, False))

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

        tainted = clean + [{"seed": RESERVED_NOT_BURNED}]
        message = _reject_reserved_seeds_in_records(tainted, False)
        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn(str(RESERVED_NOT_BURNED), message)
        # The opt-in still works FOR A RESERVED-BUT-NOT-BURNED SEED, and malformed
        # seeds must not crash the check. `FINAL_HOLDOUT_SEED_FLOOR + 7` stood here
        # until C151 burned the block it sits in; see `RESERVED_NOT_BURNED`.
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


class TheBurnedBlockAndTheOwnerRatificationTests(unittest.TestCase):
    """C151. `19,200,000`-`19,200,259` is spent; `19,300,000`-`19,300,199` is ratified.

    The reservation this file already enforced is one an opt-in can lift. The burn is
    not: the block is spent three different ways and no flag reopens it. Those are
    different properties and they are enforced by different functions, so that the
    burn's unconditionality does not depend on the order of two `if`s inside one.

    `OWNER_RATIFIED` is here for a reason that is about process rather than seeds. The
    program's rule was "only the owner can bless the replacement window", it lived in a
    document, and an agent walked past it -- `reports/c141_final_holdout_prediction.md`
    says "chosen by me rather than deferred" in the pre-registration itself, and the
    recovered disclosure says the disposition was explicitly left to the owner and
    "I have **not** chosen". Pinning the constant converts the blessing into a diff that
    carries the owner's name and cannot land without review.

    MUTATION BATTERY: 18 applied across this class and
    `TheRatifiedC151WindowIsNotYetSweptTests` in `tests/test_seed_registry_coverage.py`, 18
    caught, plus a clean-tree control that stays green. Restore was `git checkout HEAD -- .`
    with `git status --porcelain` verified empty between mutations. The ten that reach THIS
    class, and each is a real escape route rather than a typo:

      1. the burn delegation deleted from `_reject_unguarded_final_holdout`, so the opt-in
         reopens the block -> 3 red, including the end-to-end CLI pin, which ERRORS because
         `main()` then runs past the guard into a nonexistent Showdown root. That error is
         the pin working: without the guard there is nothing between argv and the sweep;
      2. the aggregation burn scan moved AFTER the `opted_in` early return -> 1 red. This is
         the fail-open the reserved-seed guard already learned once, one level up;
      3. the burned block shrunk to C141's swept span, dropping the contaminated head and
         the overrun tail -> 4 red across both modules;
      4. `OWNER_RATIFIED`'s owner rewritten from `scott` to `agent` -> 3 red. A window with
         someone else's name on it is a self-blessing with extra steps;
      5. `RATIFIED_FINAL_HOLDOUT` moved while the label stayed -> 2 red (the drift pin);
      6. the typo exemplar restored to `19,300,000` -> 1 red;
      7. the precondition dropped from the reserved-branch message -> 1 red;
      8. the overrun reason stripped from the burn message -> 1 red. "Reserved" without a
         reason is what sends a reader looking for the flag that lifts it;
      9. the recovered disclosure deleted -> 1 red here and 1 in the sibling module;
     10. and the clean-tree control, green, so none of the above is a false alarm.
    """

    def test_the_whole_burned_block_is_refused_including_both_edges(self) -> None:
        low, high = BURNED_FINAL_HOLDOUT
        self.assertEqual((low, high), (19_200_000, 19_200_259))
        for seed in (low, low + 59, low + 60, high - 1, high):
            self.assertIsNotNone(
                _reject_burned_final_holdout(seed, 1),
                f"{seed} is inside the burned block and must be refused",
            )
        # One seed either side is NOT burned. Above it is merely reserved, which is a
        # different refusal with a different message; below it is ordinary seed space.
        self.assertIsNone(_reject_burned_final_holdout(low - 1, 1))
        self.assertIsNone(_reject_burned_final_holdout(high + 1, 1))
        # And a span that only CLIPS the block is caught -- the C141 overrun shape.
        self.assertIsNotNone(_reject_burned_final_holdout(low - 50, 100))
        self.assertIsNotNone(_reject_burned_final_holdout(high - 5, 200))

    def test_the_opt_in_does_not_open_the_burned_block(self) -> None:
        # The load-bearing pin of the burn. Every one of these returned None before
        # C151, because the opt-in short-circuited the whole guard.
        low, high = BURNED_FINAL_HOLDOUT
        for seed_start, games in ((low, 60), (low + 60, 140), (high - 59, 60), (low, 260)):
            self.assertIsNotNone(
                _reject_unguarded_final_holdout(seed_start, games, True),
                f"{FINAL_HOLDOUT_OPT_IN} opened burned seeds at {seed_start}/{games}",
            )

    def test_the_burn_message_says_why_rather_than_merely_reserved(self) -> None:
        # "Reserved" is what the block was. It is now spent, three ways, and a reader
        # who is told only "reserved" will go looking for the flag that lifts it.
        message = _reject_burned_final_holdout(19_200_000, 260)
        assert message is not None
        self.assertIn("BURNED", message)
        for reason in (
            "pre-guard convenience loop",          # the contaminated head
            "chose",                               # the self-blessed C141 window
            "overrunning its registration",        # the 60-seed overrun
        ):
            self.assertIn(reason, message, f"the burn message no longer says: {reason}")
        self.assertIn(
            "reports/rust-fidelity/final_holdout_contamination_disclosure.md", message,
            "the message must point at the recovered disclosure, not just assert it",
        )
        self.assertIn(f"{FINAL_HOLDOUT_OPT_IN} does\nNOT open".replace("\n", " "), message)
        self.assertIn(OWNER_RATIFIED[0], message)

    def test_aggregating_burned_seeds_is_refused_even_with_the_opt_in(self) -> None:
        # The aggregation half. C141's committed checkpoint carries 200 of these seeds,
        # so this is a live input: without the burn scan running BEFORE the `opted_in`
        # early return, `--merge-from` plus the opt-in still writes a report that makes
        # burned seeds look measured.
        from engine_transition_differential import _reject_reserved_seeds_in_records

        low, high = BURNED_FINAL_HOLDOUT
        for opted_in in (False, True):
            for seed in (low, low + 60, high):
                message = _reject_reserved_seeds_in_records([{"seed": seed}], opted_in)
                self.assertIsNotNone(
                    message,
                    f"burned seed {seed} aggregated with opted_in={opted_in}",
                )
                assert message is not None
                self.assertIn("BURNED", message)
        # A reserved-but-not-burned seed still takes the ordinary path, so the burn did
        # not simply swallow the whole guard.
        ordinary = _reject_reserved_seeds_in_records([{"seed": RESERVED_NOT_BURNED}], False)
        assert ordinary is not None
        self.assertNotIn("BURNED", ordinary)
        self.assertIsNone(
            _reject_reserved_seeds_in_records([{"seed": RESERVED_NOT_BURNED}], True)
        )

    def test_the_owner_ratification_constant_is_exactly_what_was_signed(self) -> None:
        # Exact, both halves. A window without a name is a self-blessing with extra
        # steps, and a name without a date cannot be checked against anything.
        self.assertEqual(
            OWNER_RATIFIED, ("19,300,000-19,300,199", "scott, 2026-08-08"),
            "the owner ratification constant changed. That is a decision, not a "
            "refactor: get it ratified before editing it.",
        )

    def test_the_ratification_string_and_the_numeric_band_cannot_drift(self) -> None:
        # Two representations of one fact, so they get tied together rather than both
        # being edited by hand. This is the pin that catches "moved the band, forgot the
        # label", which is the shape of the C141 registry defect one axis over.
        low, high = (int(part.replace(",", "")) for part in OWNER_RATIFIED[0].split("-"))
        self.assertEqual((low, high), RATIFIED_FINAL_HOLDOUT)
        self.assertEqual(high - low + 1, 200, "the ratified window is 200 games")
        # Ratified, and disjoint from what was burned.
        burn_low, burn_high = BURNED_FINAL_HOLDOUT
        self.assertFalse(low <= burn_high and burn_low <= high)
        # Inside the guarded half-line, so the existing floor protects it.
        self.assertGreaterEqual(low, FINAL_HOLDOUT_SEED_FLOOR)

    def test_the_precondition_is_recorded_beside_the_window(self) -> None:
        # Ratification is not permission to run today. The trigger is not machine
        # checkable, which is exactly why it has to be written where the operator will
        # be standing when they reach for the flag.
        self.assertIn("terminal", RATIFIED_SWEEP_PRECONDITION)
        self.assertIn("frozen", RATIFIED_SWEEP_PRECONDITION)
        message = _reject_unguarded_final_holdout(RATIFIED_FINAL_HOLDOUT[0], 200, False)
        assert message is not None
        self.assertIn(RATIFIED_SWEEP_PRECONDITION, message)

    def test_the_typo_exemplar_no_longer_names_the_ratified_window(self) -> None:
        # The illustration in the guard's own comment used to be 19,300,000, which is
        # now the real target. An example that names the thing it is warning you not to
        # hit reads backwards, and a reader who greps for the window finds a comment
        # calling it a typo.
        source = (REPO / "scripts" / "engine_transition_differential.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("a typo of 19,700,000 should not", source)
        self.assertNotIn("a typo of 19,300,000 should not", source)

    def test_the_cli_refuses_the_burned_block_with_the_opt_in_passed(self) -> None:
        # END TO END, and this is the pin that matters most in this class: the helper
        # being right is worth nothing if `main()` routes around it. `--skip-build-check`
        # and a nonexistent `--showdown-root` take build freshness and dex loading out of
        # the verdict, so if the guard were anywhere after them this fails with a crash
        # rather than exit 2 -- the same discipline the reserved-seed pin above records
        # having learned the hard way.
        import tempfile

        from engine_transition_differential import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "must-not-exist.json"
            ck = Path(tmp) / "must-not-exist.jsonl"
            code = main(
                [
                    "--games", "260",
                    "--seed-start", str(BURNED_FINAL_HOLDOUT[0]),
                    "--matcher", "strict",
                    FINAL_HOLDOUT_OPT_IN,
                    "--skip-build-check",
                    "--showdown-root", "/nonexistent/holdout/guard/burned",
                    "--json", str(out),
                    "--checkpoint", str(ck),
                ]
            )
            self.assertEqual(
                code, 2,
                "the opt-in reopened the burned block through the real CLI",
            )
            self.assertFalse(out.exists(), "a refused run must leave no report")
            self.assertFalse(ck.exists(), "a refused run must leave no checkpoint")

    def test_the_recovered_disclosure_is_committed_and_says_what_the_guard_cites(self) -> None:
        # C151 recovered the disclosure the C141 pre-registration cited and that no blob
        # on any ref carried. The guard's message now points at it by path, so the path
        # has to resolve -- a citation that 404s is what created this whole incident.
        disclosure = REPO / "reports" / "rust-fidelity" / "final_holdout_contamination_disclosure.md"
        self.assertTrue(disclosure.is_file(), "the recovered disclosure is not committed")
        text = disclosure.read_text(encoding="utf-8")
        # The three facts the burn message leans on, checked against the source rather
        # than restated from it.
        self.assertIn("for start in 19100000 19000000 19200000", text)
        self.assertIn("19,200,000`–`19,200,059", text)
        self.assertIn("I have **not** chosen", text)


if __name__ == "__main__":
    unittest.main()
