"""The boundary verdict partition is FOUR-term, and this module refuses the two-term form.

`transitions_matched + transitions_diverged == boundaries_measured` was asserted as a
property of the transition differential across `reports/` and `docs/` for the whole
C111-C141 era. It is false. `boundaries_measured` increments as the last statement of
`_prepare_boundary`, so a boundary that reached the matcher is inside it; `run_game`
then routes that boundary to exactly one of FOUR counters, and two of them are neither
`matched` nor `diverged`:

    transitions_matched + transitions_diverged + engine_errors
        + counters["skip:strict_all_branches_lossy"]  ==  boundaries_measured

The two extra terms were 0 on the dev and validation-holdout windows the era iterated
against, which is the only reason the two-term form ever appeared to hold. It is broken
on artifacts that were already committed when the claim was being made
(`reports/c26_structural_probe_report.json` and `reports/c27_structural_probe_report.json`,
lossy 2 apiece) and on C141's final-holdout sweep (lossy 4, PR #1159).

WHAT THIS MODULE PINS, and why each pin is not vacuous:

1. The four-term identity closes on EVERY committed sweep artifact. Discovered by glob,
   not by a hardcoded list, so a new artifact that violates it is caught without anyone
   remembering to add it here.
2. The two-term form FAILS on the named counterexamples. This is the anti-vacuity pin:
   without it, pin 1 would pass identically on a repo where the third term never fires,
   and the whole defect is that nobody noticed that state of affairs was temporary.
3. At least one committed artifact carries `skip:strict_all_branches_lossy > 0`, so the
   corpus pin 1 runs over genuinely exercises the term.
4. `verdict_partition_failures` goes RED on a report that drops the lossy term, on a
   report whose partition over-counts, and on a report whose `boundaries_measured` is
   missing or malformed -- rather than defaulting the denominator to 0 and closing.
5. `cert_sweep_readout` actually gates on it per shard, and the failure reaches
   `gate_failures` with a nonzero exit code.

See `reports/c144_boundary_identity_correction.md`.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from engine_transition_differential import (  # noqa: E402
    VERDICT_PARTITION_COUNTERS,
    VERDICT_PARTITION_LOSSY_COUNTER,
    VERDICT_PARTITION_SCALARS,
    verdict_partition_failures,
)

# The two artifacts that were ALREADY in the repo, refuting the two-term identity, while
# the two-term identity was being asserted in report after report. Pinned by name and by
# value: if either is ever rewritten so the counterexample disappears, this module must
# fail rather than quietly become vacuous.
_COUNTEREXAMPLES = {
    "reports/c26_structural_probe_report.json": {
        "boundaries_measured": 4738,
        "transitions_matched": 4672,
        "transitions_diverged": 64,
        "engine_errors": 0,
        VERDICT_PARTITION_LOSSY_COUNTER: 2,
    },
    "reports/c27_structural_probe_report.json": {
        "boundaries_measured": 4738,
        "transitions_matched": 4676,
        "transitions_diverged": 60,
        "engine_errors": 0,
        VERDICT_PARTITION_LOSSY_COUNTER: 2,
    },
}

# C141's final-holdout sweep, the run that forced this correction. Its artifact lands with
# PR #1159 rather than with this module, so it is pinned as VALUES here and picked up by
# the glob pin once it is on main. Sourced from
# `reports/artifacts/c141_final_holdout_sweep.json` on `report-final-holdout-sweep`.
_C141 = {
    "boundaries_measured": 16274,
    "transitions_matched": 16268,
    "transitions_diverged": 2,
    "engine_errors": 0,
    "counters": {VERDICT_PARTITION_LOSSY_COUNTER: 4, "strict:lossy_render": 14},
}


# The number of committed sweep artifacts, pinned EXACTLY rather than as a floor.
#
# A floor was the one fail-open in review's ten-mutation battery. The selector below used
# to require all three of VERDICT_PARTITION_SCALARS to be present, which meant it filtered
# on exactly the keys the checker refuses: deleting `engine_errors` from
# `c134_collapsed_dev_sweep.json` dropped that artifact out of the corpus instead of
# failing it, and the whole 190-test suite stayed green -- with `> 40` against 70 artifacts
# there were 29 more it could have silently eaten. Membership is now decided by a key the
# checker does NOT validate, and the count is exact so a disappearance is a failure even if
# some future selector regains a filter.
#
# Committing a new sweep artifact means bumping this number. That is the point: it makes the
# new artifact pass through the checker deliberately rather than by default.
#
# 70 -> 71: `reports/artifacts/c141_final_holdout_sweep.json`, the C141 final-holdout sweep,
# added by PR #1159. This is NOT a routine re-stamp -- it is the sweep whose lossy count of 4
# forced this whole correction, and it is the THIRD corpus member to carry a nonzero
# `skip:strict_all_branches_lossy`, and the first in the `reports/artifacts` sweep series.
# `c26` and `c27` (lossy 2 each) were already in the corpus and are the pre-existing
# counterexamples pinned above; c141 is emphatically NOT the first firing in this repo. Of the
# 56 artifacts in `reports/artifacts/` carrying a `counters` block, 55 are at 0 and c141 alone
# is nonzero -- which is the whole and only sense in which it is a first.
#
# `_C141`'s hardcoded values above are now redundant with the glob, since the artifact is
# committed and selected; they are retained as an explicit named pin. Verified equal to the
# committed artifact at bump time: 16274 / 16268 / 2 / 0, lossy 4.
#
# PR #1159 also adds `reports/artifacts/c141_final_holdout_replay.json`, which is deliberately
# NOT in this corpus: it is a branch-level replay of two retained repros and carries no
# top-level `boundaries_measured`, so the selector below excludes it. Verified, not assumed --
# as is the fact that #1161's `c143_heal_attribution_probe.json`, which landed on main in the
# same merge, is outside the selector too. The delta really is this one sweep artifact.
#
# 71 -> 75 (C145). The four added are the one-game `19100170` bisect sweeps
# `reports/artifacts/c145_g19100170_{2ec0cb13,dc6e1e19,d27316b6,head}.json`.
#
# MEASURED, not computed. An earlier revision of this branch read 74 against a base of 70, and
# `71 + 4` happening to agree with the selector here is a coincidence of this merge, not a method:
# membership turns on a top-level `boundaries_measured` and nothing else, so a file can land in
# `reports/artifacts/` without entering this corpus -- `c145_settling_branch_dump.json` does exactly
# that, as do #1159's `c141_final_holdout_replay.json` and #1161's two `c143` artifacts noted above.
# The directory count and the corpus count therefore move independently and neither may be used to
# sanity-check the other.
#
# The procedure, which is what caught C145's bump rather than any reasoning about it:
#   1. run the selector below over BOTH trees -- the merged one and the base it came from;
#   2. confirm the set difference is exactly the files the PR adds, with NOTHING removed (a member
#      that silently stops carrying the key drops OUT, which is this pin's fail-open and which pure
#      addition would mask);
#   3. set the number to the measured count and confirm the module is green;
#   4. confirm the pin is still LIVE by setting it one lower and watching it fail -- a bump that
#      disarms this pin is indistinguishable from a correct one in the diff.
# All four c145 members close the four-term identity through the checker below: 79 + 2 + 0 + 0 == 81
# at the two pre-fix commits, 81 + 0 + 0 + 0 == 81 at the two post-fix ones.
_EXPECTED_SWEEP_ARTIFACTS = 75


def _sweep_reports() -> list[tuple[str, dict]]:
    """Every committed JSON that is shaped like a differential sweep report.

    Selected on the presence of `boundaries_measured` ALONE -- deliberately, and see
    `_EXPECTED_SWEEP_ARTIFACTS` above for why. A report missing a verdict scalar stays IN
    the corpus and is refused by `verdict_partition_failures`; it does not quietly leave.
    Verified at the time of writing: every committed JSON with a top-level
    `boundaries_measured` also carries all three scalars, so this selector picks out the
    same 70 files the stricter one did.
    """

    found: list[tuple[str, dict]] = []
    for pattern in ("reports/*.json", "reports/artifacts/*.json"):
        for path in sorted(glob.glob(os.fspath(REPO / pattern))):
            try:
                loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(loaded, dict):
                continue
            if "boundaries_measured" not in loaded:
                continue
            found.append((os.path.relpath(path, REPO), loaded))
    return found


class VerdictPartitionOverCommittedArtifactsTests(unittest.TestCase):
    def test_the_corpus_of_sweep_artifacts_is_exactly_the_expected_size(self) -> None:
        # Pin 1 below is a loop, and a loop over nothing passes. A floor is not enough
        # either: an artifact silently leaving the corpus must be a failure, not slack.
        found = _sweep_reports()
        self.assertEqual(
            len(found), _EXPECTED_SWEEP_ARTIFACTS,
            "the committed sweep-artifact corpus changed size. If an artifact was added, "
            "bump _EXPECTED_SWEEP_ARTIFACTS. If one vanished or stopped carrying a "
            "top-level boundaries_measured, that is the fail-open this pin exists to "
            f"catch. Found: {sorted(name for name, _ in found)}",
        )

    def test_every_artifact_in_the_corpus_carries_every_verdict_scalar(self) -> None:
        # The selector no longer filters on these, so their absence has to be asserted
        # somewhere. Here, plus the checker itself, which refuses rather than skips.
        for name, report in _sweep_reports():
            with self.subTest(artifact=name):
                for field in VERDICT_PARTITION_SCALARS:
                    self.assertIn(
                        field, report,
                        f"{name}: a sweep report without {field} cannot be adjudicated, "
                        "and must not be able to leave the corpus by omitting it",
                    )

    def test_the_counter_and_scalar_vocabularies_agree_on_every_artifact(self) -> None:
        # VERDICT_PARTITION_COUNTERS' job. The two tuples name the SAME partition in two
        # vocabularies -- `run_game`'s internal counter keys and the report's published
        # scalars -- and `checkpoint_report_binding_failures` only binds them on the
        # checkpoint-merge path, so a plain shard's binding is otherwise unchecked.
        for name, report in _sweep_reports():
            counters = report.get("counters")
            if not isinstance(counters, dict):
                continue
            with self.subTest(artifact=name):
                for scalar, counter in zip(
                    VERDICT_PARTITION_SCALARS, VERDICT_PARTITION_COUNTERS
                ):
                    # A bare KeyError here would still be red, but it would name the
                    # missing key and not the drift. Fail with the message instead.
                    self.assertIn(scalar, report, f"{name}: no published {scalar}")
                    self.assertEqual(
                        report[scalar], counters.get(counter, 0),
                        f"{name}: published {scalar} disagrees with counters[{counter!r}], "
                        "so the two vocabularies for one partition have drifted",
                    )

    def test_the_four_term_partition_closes_on_every_committed_artifact(self) -> None:
        for name, report in _sweep_reports():
            with self.subTest(artifact=name):
                self.assertEqual(
                    verdict_partition_failures(report, label=name), [],
                    f"{name}: the four-term boundary verdict partition does not close",
                )

    def test_the_two_term_form_is_refuted_by_artifacts_already_in_the_repo(self) -> None:
        # THE ANTI-VACUITY PIN. Pin 1 above would pass unchanged on a repo where the
        # lossy verdict never fires, and that is exactly the repo the false claim was
        # made in. These two artifacts were committed BEFORE the claim was last
        # repeated, so the counterexample was always available to anyone who looked.
        for name, expected in _COUNTEREXAMPLES.items():
            with self.subTest(artifact=name):
                report = json.loads((REPO / name).read_text(encoding="utf-8"))
                measured = report["boundaries_measured"]
                matched = report["transitions_matched"]
                diverged = report["transitions_diverged"]
                lossy = report["counters"][VERDICT_PARTITION_LOSSY_COUNTER]
                for key, value in expected.items():
                    actual = (
                        report["counters"][key]
                        if key == VERDICT_PARTITION_LOSSY_COUNTER
                        else report[key]
                    )
                    self.assertEqual(
                        actual, value,
                        f"{name}: {key} moved; if this artifact was legitimately "
                        "re-measured, re-derive the counterexample rather than "
                        "deleting this pin",
                    )
                self.assertGreater(lossy, 0, f"{name}: expected a live lossy verdict")
                # The false invariant, shown false.
                self.assertNotEqual(
                    matched + diverged, measured,
                    f"{name}: the two-term identity holds here, so this artifact is no "
                    "longer a counterexample and C144's evidence has evaporated",
                )
                # The true one, shown true, on the same numbers.
                self.assertEqual(
                    matched + diverged + report["engine_errors"] + lossy, measured,
                    f"{name}: the four-term identity does not close either",
                )

    def test_some_committed_artifact_actually_exercises_the_lossy_verdict(self) -> None:
        live = [
            name
            for name, report in _sweep_reports()
            if (report.get("counters") or {}).get(VERDICT_PARTITION_LOSSY_COUNTER, 0) > 0
        ]
        self.assertTrue(
            live,
            "no committed artifact has skip:strict_all_branches_lossy > 0, so the "
            "third term of the identity is untested by real data",
        )

    def test_c141_final_holdout_numbers_break_two_terms_and_close_on_four(self) -> None:
        # The run that forced the correction. Pinned as values because its artifact
        # arrives on a different PR; once merged the glob pin covers it too.
        report = dict(_C141)
        self.assertNotEqual(
            report["transitions_matched"] + report["transitions_diverged"],
            report["boundaries_measured"],
        )
        self.assertEqual(verdict_partition_failures(report, label="c141"), [])
        # `strict:lossy_render` is PER BRANCH and must never be folded into the
        # identity: 14 lossy branch renders produced only 4 unadjudicable boundaries.
        self.assertGreater(
            report["counters"]["strict:lossy_render"],
            report["counters"][VERDICT_PARTITION_LOSSY_COUNTER],
        )


class VerdictPartitionFailuresTests(unittest.TestCase):
    """The checker must be able to go red. Each case below is a distinct way in."""

    def _report(self, **overrides) -> dict:
        report = {
            "boundaries_measured": 100,
            "transitions_matched": 95,
            "transitions_diverged": 3,
            "engine_errors": 1,
            "counters": {VERDICT_PARTITION_LOSSY_COUNTER: 1},
        }
        report.update(overrides)
        return report

    def test_a_closing_partition_reports_nothing(self) -> None:
        self.assertEqual(verdict_partition_failures(self._report()), [])

    def test_a_denominator_derived_from_the_two_term_form_is_caught(self) -> None:
        # THE C144 DEFECT, mechanized. A writer who believes `measured == matched +
        # diverged` computes a denominator that excludes the lossy and engine_error
        # verdicts, so the verdict tally exceeds it.
        failures = verdict_partition_failures(self._report(boundaries_measured=98))
        self.assertEqual(len(failures), 1)
        self.assertIn("does not close", failures[0])
        self.assertIn("2 more verdicts than measured boundaries", failures[0])

    def test_an_uncounted_post_measurement_exit_is_caught_in_the_other_direction(self) -> None:
        # The refactor risk: `run_game` grows a fifth way out of the matcher and nobody
        # counts it, so measured boundaries acquire no verdict at all.
        failures = verdict_partition_failures(self._report(transitions_matched=93))
        self.assertEqual(len(failures), 1)
        self.assertIn("2 measured boundaries carry no verdict", failures[0])

    def test_a_report_with_no_lossy_counter_defaults_it_to_zero(self) -> None:
        # A Counter dump omits unseen keys, and a non-strict matcher never emits this
        # one, so absence must read as 0 -- but ONLY for this term.
        self.assertEqual(
            verdict_partition_failures(
                self._report(
                    boundaries_measured=99, transitions_matched=95, counters={},
                )
            ),
            [],
        )

    def test_a_missing_denominator_is_refused_rather_than_defaulted(self) -> None:
        # Defaulting `boundaries_measured` to 0 would make the identity close on an
        # unreadable report. That is the instrument-that-cannot-move failure, so it
        # must be a failure and not a pass.
        for bad in ({}, {"boundaries_measured": None}, {"boundaries_measured": "100"},
                    {"boundaries_measured": -1}, {"boundaries_measured": True}):
            with self.subTest(bad=bad):
                report = self._report()
                report.pop("boundaries_measured")
                report.update(bad)
                failures = verdict_partition_failures(report)
                self.assertTrue(failures)
                self.assertIn("boundaries_measured", failures[0])

    def test_a_malformed_verdict_scalar_is_refused(self) -> None:
        for field in VERDICT_PARTITION_SCALARS:
            with self.subTest(field=field):
                failures = verdict_partition_failures(self._report(**{field: "3"}))
                self.assertTrue(any(field in text for text in failures))

    def test_a_missing_counters_block_is_refused(self) -> None:
        failures = verdict_partition_failures(self._report(counters=None))
        self.assertTrue(any("counters" in text for text in failures))

    def test_the_counter_vocabulary_matches_the_scalar_vocabulary_in_order(self) -> None:
        # The two name tuples describe the same partition. If a future edit adds a fifth
        # verdict to one and not the other, the identity silently splits in two.
        self.assertEqual(
            len(VERDICT_PARTITION_COUNTERS),
            len(VERDICT_PARTITION_SCALARS) + 1,
        )
        self.assertEqual(VERDICT_PARTITION_COUNTERS[-1], VERDICT_PARTITION_LOSSY_COUNTER)


class ReadoutGatesOnThePartitionTests(unittest.TestCase):
    """The check has to be WIRED IN, not merely available."""

    def _run(self, shard_overrides: dict) -> dict:
        import cert_sweep_readout as readout

        payload = {
            "boundaries_measured": 100,
            "boundaries_full_round": 100,
            "transitions_matched": 99,
            "transitions_diverged": 0,
            "engine_errors": 0,
            "games": 1,
            "counters": {VERDICT_PARTITION_LOSSY_COUNTER: 1},
            "divergence_classes": {},
            "repros": [],
            "repro_retention": {
                "repros_complete": True,
                "repros_retained": 0,
                "transitions_diverged": 0,
            },
            "build_check": "gated",
            "matcher": "strict",
            "seeds": {"min": 0, "max": 0, "distinct": 1},
        }
        payload.update(shard_overrides)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard-0.json"
            shard.write_text(json.dumps(payload), encoding="utf-8")
            prediction = root / "prediction.json"
            prediction.write_text(
                json.dumps({"predicted_class_rates_10k": {}}), encoding="utf-8"
            )
            output = root / "out.json"
            code = readout.main([
                "--shards", str(shard),
                "--prediction", str(prediction),
                "--json", str(output),
            ])
            return {"code": code, "out": json.loads(output.read_text(encoding="utf-8"))}

    def test_the_partition_is_the_only_thing_the_broken_shard_adds(self) -> None:
        # A DIFFERENTIAL, not a bare assertion. These fixtures already fail the
        # certification contract for unrelated reasons, so `code != 0` on the broken
        # shard proves nothing. What proves the gate fired is that the set of
        # gate_failures grows by exactly the partition failure and nothing else.
        clean = set(self._run({})["out"]["gate_failures"])
        broken = set(self._run({"boundaries_measured": 98})["out"]["gate_failures"])
        added = broken - clean
        self.assertEqual(len(added), 1, added)
        only = added.pop()
        self.assertIn("shard-0.json", only)
        self.assertIn("verdict partition does not close", only)
        self.assertEqual(clean - broken, set())
        self.assertEqual([f for f in clean if "verdict partition" in f], [])

    def test_the_gate_is_per_shard_so_opposite_violations_cannot_cancel(self) -> None:
        # Two shards that break the partition in opposite directions sum to a clean
        # aggregate. A gate on the aggregate would pass; this one must not.
        import cert_sweep_readout as readout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, measured in ((0, 99), (1, 101)):
                payload = {
                    "boundaries_measured": measured,
                    "boundaries_full_round": 100,
                    "transitions_matched": 100,
                    "transitions_diverged": 0,
                    "engine_errors": 0,
                    "games": 1,
                    "counters": {},
                    "divergence_classes": {},
                    "repros": [],
                    "repro_retention": {
                        "repros_complete": True,
                        "repros_retained": 0,
                        "transitions_diverged": 0,
                    },
                    "build_check": "gated",
                    "matcher": "strict",
                    "seeds": {"min": index, "max": index, "distinct": 1},
                }
                path = root / f"shard-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(str(path))
            prediction = root / "prediction.json"
            prediction.write_text(
                json.dumps({"predicted_class_rates_10k": {}}), encoding="utf-8"
            )
            output = root / "out.json"
            readout.main(
                ["--shards", *paths, "--prediction", str(prediction), "--json", str(output)]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        aggregate = payload["aggregate"]
        self.assertEqual(
            aggregate["transitions_matched"] + aggregate["transitions_diverged"],
            aggregate["boundaries_measured"],
            "the aggregate must LOOK clean, or this test is not testing cancellation",
        )
        matching = [f for f in payload["gate_failures"] if "verdict partition" in f]
        self.assertEqual(len(matching), 2, payload["gate_failures"])


class DifferentialSelfCheckIsArmedTests(unittest.TestCase):
    """The differential's own two exit paths must gate on the partition.

    A SOURCE pin, and said plainly: the behavioural path costs a real sweep or a
    hand-built checkpoint merge, so this is a guard against the check being deleted, not
    evidence that it fires. It goes red if either call site or either exit-code term is
    removed, which is the "guard nothing pins" failure this repo has been blocked for.
    Parsed with `ast` rather than grepped, so reformatting does not break it and deletion
    does.
    """

    def _main(self):
        import ast

        source = (REPO / "scripts" / "engine_transition_differential.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                return node
        self.fail("engine_transition_differential.main not found")

    def test_both_report_paths_compute_the_partition(self) -> None:
        import ast

        calls = [
            node
            for node in ast.walk(self._main())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "verdict_partition_failures"
        ]
        self.assertEqual(
            len(calls), 2,
            "main() has two report-emitting paths (the --merge-from path and the sweep "
            "path) and each must self-check its own report",
        )

    def test_the_seam_that_hides_the_fifth_path_is_still_in_place(self) -> None:
        """The identity survives the fifth path by an accident. Pin the accident.

        Between the `boundaries_measured` increment and the verdict there is a stretch of
        `run_game` outside the matcher's `try` -- `env.step`, `_fold`, the `active_changed`
        comprehension, the re-raised KeyboardInterrupt/SystemExit, `classify_divergence`.
        An escape from any of those leaves the boundary measured with NO verdict: a real
        fifth outcome that nothing counts.

        It is not a partition violation only because (a) `counts` is a LOCAL Counter in
        `run_game`, not a caller-owned one, and (b) the sweep loop does not catch
        exceptions from `run_game`, so a crashing game's counts are discarded wholesale.
        Take either away -- most plausibly by "salvaging partial counts so a long sweep
        does not lose a game" -- and the identity is five-term instantly.

        So both halves of the accident are asserted here rather than left as a comment.
        This is a SOURCE pin and does not prove the escape is unreachable; it proves the
        two conditions that currently make it harmless have not been removed silently.
        """
        import ast

        source = (REPO / "scripts" / "engine_transition_differential.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        run_game = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_game"
        )
        # (a) `counts` is created inside run_game and is not a parameter.
        params = {
            arg.arg for arg in run_game.args.args + run_game.args.kwonlyargs
        }
        self.assertNotIn(
            "counts", params,
            "run_game now takes `counts` from its caller, so a crashing game's partial "
            "counts can outlive it. Count the post-measurement escape explicitly and add "
            "it to the verdict partition -- see C144 §2a.",
        )
        local = [
            node for node in run_game.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "counts"
        ]
        self.assertEqual(
            len(local), 1,
            "run_game must create its own `counts`; that locality is what discards the "
            "increment when a game raises after being measured (C144 §2a)",
        )

        # (b) the sweep loop does not CATCH exceptions from run_game. A bare try/finally
        # is fine and is what is there today (it closes the checkpoint handle); a try with
        # `except` handlers around the call is what would salvage the counts.
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        def contains(node, target) -> bool:
            return any(child is target or contains(child, target)
                       for child in ast.iter_child_nodes(node))

        calls = [
            node for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_game"
        ]
        self.assertEqual(len(calls), 1, "expected exactly one run_game call site in main")
        call = calls[0]
        catching = [
            node for node in ast.walk(main)
            if isinstance(node, ast.Try)
            and node.handlers
            and any(contains(stmt, call) or stmt is call for stmt in node.body)
        ]
        self.assertEqual(
            catching, [],
            "the run_game call is now inside a try/except, so a game that raises AFTER "
            "boundaries_measured incremented can have its partial counts recorded. That "
            "makes the boundary verdict partition five-term. Count the escape (e.g. "
            "`boundary_abandoned_after_measure`) and add it to VERDICT_PARTITION_SCALARS "
            "or VERDICT_PARTITION_COUNTERS -- see C144 §2a.",
        )

    def test_both_report_paths_fold_the_partition_into_the_exit_code(self) -> None:
        import ast

        gated = 0
        for node in ast.walk(self._main()):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            names = {
                child.id for child in ast.walk(node.value) if isinstance(child, ast.Name)
            }
            if "report" in names and "partition" in names:
                gated += 1
        self.assertEqual(
            gated, 2,
            "an exit code that ignores the partition means a report whose counters do "
            "not account for every measured boundary still exits 0",
        )


class FidelityDenominatorTests(unittest.TestCase):
    """`in_support_rate` must not credit an unadjudicated boundary as in-support.

    The old formula was `(boundaries_measured - transitions_diverged) / boundaries_measured`,
    which is the two-term identity wearing a different hat: it assumes every measured
    boundary that did not diverge matched. A boundary whose every branch rendered lossy
    answered nothing, so it leaves the denominator.
    """

    def _fidelity(self, *, measured: int, matched: int, diverged: int, lossy: int,
                  engine_errors: int = 0) -> dict:
        import cert_sweep_readout as readout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard-0.json"
            shard.write_text(json.dumps({
                "boundaries_measured": measured,
                "boundaries_full_round": measured,
                "transitions_matched": matched,
                "transitions_diverged": diverged,
                "engine_errors": engine_errors,
                "games": 100,
                "counters": {VERDICT_PARTITION_LOSSY_COUNTER: lossy},
                "divergence_classes": {},
                "repros": [],
                "repro_retention": {
                    "repros_complete": True,
                    "repros_retained": 0,
                    "transitions_diverged": diverged,
                },
                "build_check": "gated",
                "matcher": "strict",
                "seeds": {"min": 0, "max": 99, "distinct": 100},
            }), encoding="utf-8")
            prediction = root / "prediction.json"
            prediction.write_text(
                json.dumps({"predicted_class_rates_10k": {}}), encoding="utf-8"
            )
            output = root / "out.json"
            readout.main([
                "--shards", str(shard), "--prediction", str(prediction),
                "--json", str(output),
            ])
            return json.loads(output.read_text(encoding="utf-8"))["fidelity"]

    def test_unadjudicated_boundaries_leave_the_denominator(self) -> None:
        fidelity = self._fidelity(measured=1000, matched=989, diverged=10, lossy=1)
        self.assertEqual(fidelity["boundaries_adjudicated"], 999)
        self.assertEqual(fidelity["boundaries_unadjudicated"], 1)
        self.assertEqual(fidelity["in_support_rate"], round(989 / 999, 6))
        # The old formula, for contrast: it would have read 0.99, crediting the
        # unadjudicable boundary as in-support.
        self.assertNotEqual(fidelity["in_support_rate"], 0.99)

    def test_engine_errors_are_unadjudicated_too(self) -> None:
        fidelity = self._fidelity(
            measured=1000, matched=987, diverged=10, lossy=1, engine_errors=2
        )
        self.assertEqual(fidelity["boundaries_unadjudicated"], 3)
        self.assertEqual(fidelity["in_support_rate"], round(987 / 997, 6))

    def test_a_fully_adjudicated_run_is_unchanged_by_the_correction(self) -> None:
        fidelity = self._fidelity(measured=1000, matched=990, diverged=10, lossy=0)
        self.assertEqual(fidelity["in_support_rate"], 0.99)
        self.assertEqual(fidelity["boundaries_unadjudicated"], 0)


if __name__ == "__main__":
    unittest.main()
