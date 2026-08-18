#!/usr/bin/env python3
"""The census gate must RUN, must be able to FAIL, and its failure must reach the required check.

WHY THIS FILE EXISTS. `scripts/schema_container_census.py` was a correct gate that nothing
executed. Its only caller was `scripts/schema_rotation_drill.sh:667`, and that script's only caller
is a person with 17-70 minutes to spend, so nothing ran it: the census went red at e496e9b8, when
the v4 default rotation moved a container's line, and stayed red across EIGHT consecutive
first-parent commits on main -- through SEVEN subsequent PR merges -- until #1276 re-keyed the row.
No check reported it once.

So there are three separable things to pin, and pinning fewer would leave the same hole in a new
place:

  1. STRUCTURE -- the job exists, is unconditional, and cannot be skipped or made advisory.
  2. CONSUMPTION -- `gate-status`, the one required context, demands its success and actually
     exits nonzero on it. A job whose result nothing reads is not meaningfully different from a
     job nothing runs; this repo has found FOUR inert pins, the last one exactly that shape.
  3. BEHAVIOUR -- the census still aborts on each thing it claims to catch. Structure and
     consumption are both satisfied by a census that returns 0 unconditionally.

...AND A FOURTH, WHICH THIS FILE ITSELF FAILED AT 8d32dcb2, in the same shape as its subject. This
module executed NOWHERE. `grep -rn test_schema_container_census_wiring .github/ scripts/` returned
zero hits at that commit; the workflow names test modules ONE BY ONE (34 executable `-m unittest`
invocation sites once this one is added, derived by test_unreachable_readjudication's scan, with no
`unittest discover` and no pytest step in .github/), so a new module is invisible to CI by default
and all three pins above were local-only. A job-level
`if: false`, a `continue-on-error`, removal from `gate-status`'s `needs:`, an `exit 1` flipped to
`exit 0`, an emptied pin table and `fetch-depth: 2` -> `1` would each have landed on main GREEN --
a correct guard nothing invokes, which is precisely what this file exists to forbid.

  4. EXECUTION -- the `The census wiring pins` step of that same job runs THIS module, and
     `test_the_workflow_test_count_guard_matches_this_module` derives its exact-count guard from
     the loader so the guard cannot go stale without a LOCAL failure. That self-check is what makes
     the omission impossible for this module rather than merely fixed once;
     tests/test_harness_digest_provenance.py, the precedent this file is modelled on, has carried it
     since #1163 and copying the pins without it is how the property went missing.

The behavioural half runs against MINIMAL synthetic trees rather than the repo, so each assertion
has a known answer that does not move when a container is added to src/. The class pin is
substituted per fixture for the same reason -- these tests are about the mechanism, not about the
current contents of the spec file.

⚠ THE SAME BOUNDARY HAS MOVED FIVE TIMES IN THIS PR, ONE FRAME PER ROUND, AND EVERY FIX BUT THE
LAST PINNED THE FRAME RATHER THAN THE PROPERTY. The record, because the pattern is the finding:

  * the call site -- `assertIn("fetch-depth: 2", block)` was satisfied by the step's own COMMENT;
    fixed by `_job_code` + `_has_key_line`.
  * whole-line matching -- `assertIn('rc=$?', block)` was satisfied by the kill-confirm step's own
    `rc=$?`; fixed by `_has_line`.
  * step scoping -- the same assertion over the whole job could not tell the two apart even then;
    fixed by `_step_block`.
  * ADJACENCY -- with all three in place, ONE diagnostic `echo` inserted between the census
    invocation and a standalone `rc=$?` gave `STEP EXIT=0` on a census that exited 15, with 22/22
    green. Every pin was textual; the property is runtime.
  * A REGEX OVER EXIT STATEMENTS -- `_first_exit`, written in the same commit as the fix above to
    hold the three shell regions that fix did not reach. It collects `re.fullmatch(r"exit \\d+",
    ...)`, so `exit 0 # r4`, `true && exit 0`, `exit 0;`, `trap 'exit 0' EXIT`, a call to a function
    that exits, and wrapping the real `exit 1` in `if false; then ... fi` are all invisible to it.
    Each was 26/26 green, and the first made `gate-status` -- the ONLY required context -- exit 0
    over a red census. `gate-status` was green on that commit itself, produced by the one arm
    nothing executed.

So the fix is not a sixth textual pin, and it is not a better regex either: it is the fix round 4
already got right, applied to the OTHER THREE `run:` bodies. `TheStepsBehaveWhenExecutedTests`
extracts each step's own `run:` body and RUNS it -- against a stub whose status and output it
chooses for the two census steps, a stub that answers per scratch case for the kill-confirm, a
throwaway git repo with a real merge commit for the merge-parent step, and seven bound
`needs.*.result` names for `gate-status`. Its docstring answers the question the earlier four pins
never asked about themselves: what is the next frame out, and what would have to change for the pin
to hold while the property is false. `_first_exit` is kept as belt, and documented as belt.

Two further seams closed in the same round, both measured green beforehand: the module-name
assertion and the capture-line assertion were UNLINKED, so substituting `tests.test_linear_policy`
(26 tests, stdlib only) while leaving the old string in a no-op line ran a module with no opinion
about this job; and the top-level `defaults:` prohibition was a raw substring that `defaults :`
walked past.

Round 4 also found the workflow-side guards forgeable in two ways that KEEP the real invocation
(a swallowed status plus an appended summary, and an env-gated wrapper), and the self-check
job-scoped where it had to be step-scoped. Both are recorded at their assertions.
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "engine-fidelity-gates.yml"
CENSUS = REPO / "scripts" / "schema_container_census.py"
SPEC = REPO / "tests" / "data" / "schema_drill_schema_containers.txt"
JOB = "schema-container-census"
VERDICT_STEP = "Schema-container census"
KILL_CONFIRM_STEP = "The census reddens on known-bad input"
WIRING_STEP = "The census wiring pins"
MERGE_STEP = "The tree censused is the MERGE result, not the branch tip"
GATE_JOB = "gate-status"
GATE_STEP = "Report the fidelity gate result"

#: This module's own dotted name, DERIVED rather than spelled. The step that runs it must name
#: THIS module, and a literal here would have to be kept in step with the filename by hand --
#: which is the kind of hand-maintained agreement `test_the_workflow_test_count_guard_matches_
#: this_module` exists to remove for the count.
MODULE = f"tests.{pathlib.Path(__file__).stem}"

#: The census invocation AND the capture of its status, on one line. Pinned as one string because
#: splitting them is the defect: see `TheStepsBehaveWhenExecutedTests`.
INVOKE_AND_CAPTURE = "python scripts/schema_container_census.py || rc=$?"

_PIN_NAME = "EXPECTED_CLASS_COUNTS_BY_FILE"
_TIMEOUT = 120


def _job_block(name: str) -> str:
    """The YAML text of one top-level job, without importing a YAML parser.

    `setup-python` on a bare checkout has no PyYAML, and installing one to read a file this test
    could read directly would be the kind of dependency that turns a fast unconditional job into a
    skippable one. Same reasoning, and same implementation, as
    tests/test_harness_digest_provenance.py.
    """

    text = WORKFLOW.read_text(encoding="utf-8")
    marker = f"\n  {name}:\n"
    rest = text[text.index(marker) + len(marker) :]
    lines: list[str] = []
    for line in rest.splitlines():
        # A job's body is indented four spaces or more. Anything at exactly two -- the next job's
        # key, or the comment block introducing it -- ends this one.
        if line.startswith("  ") and not line.startswith("   "):
            break
        lines.append(line)
    return "\n".join(lines)


def _job_code(name: str) -> str:
    """`_job_block` with COMMENT LINES REMOVED.

    Every content assertion below must read this and not the raw block. Found by mutation-testing
    this file: `assertIn("fetch-depth: 2", block)` was satisfied by the step's own COMMENT saying
    "`fetch-depth: 2` above is what makes demanding the parent safe", so changing the real key to
    `fetch-depth: 1` -- reintroducing the false red this job already suffered once -- left the test
    green. A guard whose pattern matches its own prose is checking nothing about its subject.
    """

    return "\n".join(
        line for line in _job_block(name).splitlines() if not line.strip().startswith("#")
    )


def _has_key_line(block: str, key: str, value: str) -> bool:
    """True when some line IS exactly `key: value`, rather than merely mentioning it."""

    return any(line.strip() == f"{key}: {value}" for line in block.splitlines())


def _has_line(block: str, text: str) -> bool:
    """True when some line IS exactly `text` once stripped, rather than merely containing it.

    The `_has_key_line` remedy generalized, and it is needed for the same reason at a different
    granularity. `assertIn('rc=$?', block)` over the whole job was satisfied by a DIFFERENT
    occurrence: the kill-confirm step's `expect()` has its own `rc=$?`, so deleting the census
    step's -- which is what makes the exit code the verdict -- left the test green. Found by
    mutation, like the `fetch-depth` case. Whole-line matching alone does not fix that one, because
    both occurrences strip to the same text; it has to be combined with `_step_block` so the
    assertion reads only the step whose property is being pinned.
    """

    return any(line.strip() == text for line in block.splitlines())


def _keys(block: str) -> list[str]:
    """Every YAML key in `block`, NORMALIZED, so a spelling variant cannot hide one.

    `assertNotIn("\\ndefaults:", text)` was the previous form of the top-level `defaults:`
    prohibition and `defaults :` -- a space before the colon, which YAML accepts -- walked
    straight past it. The same class of miss as `if : false`, which
    `test_the_job_is_unconditional` already normalizes against; recorded there and not applied
    here until review pointed at the second copy. Raw-substring key prohibitions are a defect
    shape in this file, so they are all routed through this one helper.
    """

    out: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:]
        out.append(stripped.split(":")[0].strip().strip("\"'"))
    return out


def _top_level_keys() -> list[str]:
    """Normalized keys at column 0 of the workflow -- `on`, `jobs`, and what must not join them."""

    text = WORKFLOW.read_text(encoding="utf-8")
    return _keys("\n".join(line for line in text.splitlines() if line[:1] not in ("", " ", "#")))


def _step_block(job: str, step_name: str, *, keep_comments: bool = False) -> str:
    """The YAML text of ONE named step of a job, comments removed unless `keep_comments`.

    Assertions about a step's body must not be satisfiable by a sibling step's body. Two of this
    file's assertions were, before this helper existed: see `_has_line`.

    `keep_comments` exists for `_step_script`, which EXECUTES the extracted text: a `#` line inside
    a `run: |` block is a shell comment and part of the script, so dropping it there would mean
    running something the runner does not run.
    """

    lines = _job_block(job).splitlines()
    want = f"- name: {step_name}"
    start = indent = None
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("- ") and stripped == want:
            start, indent = index, len(line) - len(stripped)
            break
    if start is None:
        raise AssertionError(f"no step named {step_name!r} in job {job!r}")
    body = [lines[start]]
    for line in lines[start + 1 :]:
        stripped = line.lstrip()
        # The next step at the SAME indentation ends this one. A deeper `- ` is a list item inside
        # this step's body (or inside a `run:` script) and must not terminate it.
        if stripped.startswith("- ") and line and (len(line) - len(stripped)) <= indent:
            break
        body.append(line)
    if keep_comments:
        return "\n".join(body)
    return "\n".join(line for line in body if not line.strip().startswith("#"))


def _step_script(job: str, step_name: str) -> str:
    """The shell script GitHub hands to `bash` for one step, dedented and comments intact.

    THE POINT OF THIS HELPER IS THAT ITS RESULT IS EXECUTED, NOT PATTERN-MATCHED. The same textual
    pin in this file has been displaced FOUR times -- call site, then whole-line matching, then step
    scoping, then ADJACENCY -- and each fix pinned the new frame rather than the property. A
    pin that runs the step and reads its exit status cannot be talked past by rearranging the text,
    because it does not care what the text looks like.

    Only the literal block form `run: |` is accepted. A folded `run: >` joins lines and would change
    the script's meaning, so it must be a loud extraction failure rather than a quiet mismatch
    between what this pin executes and what the runner executes.
    """

    lines = _step_block(job, step_name, keep_comments=True).splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "run: |":
            continue
        indent = len(line) - len(line.lstrip())
        body: list[str] = []
        for following in lines[index + 1 :]:
            if following.strip() and (len(following) - len(following.lstrip())) <= indent:
                break
            body.append(following)
        script = textwrap.dedent("\n".join(body))
        if not script.strip():
            raise AssertionError(f"the `run:` body of step {step_name!r} extracted empty")
        return script + "\n"
    raise AssertionError(f"step {step_name!r} of job {job!r} has no literal `run: |` block")


def _step_env_names(job: str, step_name: str) -> set[str]:
    """The names a step's YAML `env:` block binds.

    `_step_script` extracts the `run:` body and NOTHING ELSE, so a step whose script reads
    `${CENSUS}` under `set -u` is only executable here if this test supplies that name itself.
    That makes the fixture's env a second hand-maintained agreement, and the executed pins would
    go quietly green-on-a-green-control if a name were renamed on both sides of the fixture while
    the YAML kept the old one. Derived and compared instead: see
    `test_the_executed_gate_status_fixture_binds_the_names_the_step_declares`.
    """

    lines = _step_block(job, step_name).splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "env:":
            continue
        indent = len(line) - len(line.lstrip())
        names = set()
        for following in lines[index + 1 :]:
            if not following.strip():
                continue
            if (len(following) - len(following.lstrip())) <= indent:
                break
            names.add(following.strip().split(":")[0].strip().strip("\"'"))
        return names
    return set()


def _guard_body(block: str, needle: str) -> str:
    """The shell `if ...; then ... fi` region containing `needle`.

    Used to assert a guard's failure arm actually exits nonzero. A structural check that the guard
    EXISTS says nothing about whether it does anything.
    """

    lines = block.splitlines()
    for index, line in enumerate(lines):
        if needle not in line:
            continue
        body = [line]
        for following in lines[index + 1 :]:
            body.append(following)
            if following.strip() == "fi":
                break
        return "\n".join(body)
    raise AssertionError(f"no guard containing {needle!r} in this block")


def _first_exit(block: str, needle: str) -> str:
    """The FIRST `exit N` statement of the guard body containing `needle`.

    `assertIn("exit 1", _guard_body(...))` is satisfied by a body that exits 0 first and
    never reaches the `exit 1` -- found by this PR's round-4 battery, which inserted
    `echo "::notice::..."; exit 0` above the merge-parent arm's untouched `exit 1` and was
    NOT CAUGHT. That is the same shape as every other displacement in this file: the pin
    named a token the mutation left in place. Asserting the first exit is the one that
    decides pins the arm's verdict rather than its vocabulary.

    ⚠ AND THIS HELPER IS BELT, NOT THE PIN -- which is the round-5 finding and the reason it is
    documented here rather than hardened. It was written as a SIXTH textual pin, to hold the three
    regions round 4's `_execute` did not reach, and `re.fullmatch(r"exit \\d+", ...)` means every
    exit that is not a bare `exit N` line is invisible to it. All six of these were measured
    26/26 GREEN, and `gate-status` -- the sole required context -- exited 0 over a red census on
    the first of them:

      * `exit 0 # r4` (a trailing comment)      * `true && exit 0`
      * `exit 0;` (a trailing semicolon)        * `trap 'exit 0' EXIT`
      * a helper function whose body exits, called from the arm
      * `if false; then exit 1; fi` -- a pure restructuring with NO `exit 0` anywhere, which no
        regex over exit statements can see, because the mutation adds none

    Hardening the pattern would have been the SEVENTH pin in the sequence call site -> whole line
    -> step scope -> adjacency -> regex, and each of the first four was displaced by the next
    mutation. So the remedy is the same one round 4 got right and applied to only two of the five
    regions: EXECUTE the arm. `TheStepsBehaveWhenExecutedTests` now runs all five, and every
    bypass above is red there whatever it looks like. This helper stays because a first-exit
    assertion names the arm a reader is looking for, and reads it in the YAML where the mistake
    would be made -- but nothing here is load-bearing any more.
    """

    exits = [
        line.strip()
        for line in _guard_body(block, needle).splitlines()
        if re.fullmatch(r"exit \d+", line.strip())
    ]
    return exits[0] if exits else "<no exit at all>"


class TheCensusJobIsWiredAndCannotBeMadeAdvisoryTests(unittest.TestCase):
    def test_the_workflow_defines_the_job(self) -> None:
        self.assertIn(f"\n  {JOB}:\n", WORKFLOW.read_text(encoding="utf-8"))

    def test_the_job_is_unconditional(self) -> None:
        """No `if:`, `continue-on-error:` or `needs:` at ANY depth.

        `skipped` must not be reachable, which is what lets `gate-status` demand the positive
        instead of excluding a stale list of negatives. The key scan is normalized before
        comparison because `if : false` and `"if": false` are both valid YAML that a naive
        `"if:" in text` search slips past -- three revisions of the equivalent check in
        tests/test_harness_digest_provenance.py were each defeated by the next mutation, and the
        recorded lesson is reused here rather than rediscovered.

        `continue-on-error: true` is forbidden for the same reason as `if`: it is documented to
        stop a failing job from failing the workflow, so the census could go red while the
        required context still passed.
        """

        keys = _keys(_job_block(JOB))

        self.assertNotIn("if", keys, "the job or one of its steps became skippable")
        self.assertNotIn(
            "continue-on-error",
            keys,
            "a failing census would no longer fail the job, so gate-status would still pass",
        )
        self.assertNotIn("needs", keys, "the job now depends on another job and can be skipped")
        # Anti-vacuity, naming a key at each depth the scan claims to reach.
        self.assertIn("runs-on", keys, "anti-vacuity: the job-level key scan found nothing")
        self.assertIn("name", keys, "anti-vacuity: the step-level key scan found nothing")
        self.assertIn("run", keys, "anti-vacuity: the step-body key scan found nothing")

    def test_required_gate_status_cannot_hide_a_red_census(self) -> None:
        """The required reporter must run, and its reporting step must not be made advisory.

        The census job's own guard is insufficient: branch protection reads `gate-status`, so a
        job-level `continue-on-error`, a non-`always()` job condition, or either key on the sole
        reporting step can make a red census look green. `if: always()` is intentionally required
        at the job level -- it keeps the reporter alive to report failed dependencies -- so this
        test distinguishes that one allowed `if` from the forbidden bypasses rather than applying
        the census job's blanket prohibition.
        """

        gate_code = _job_code(GATE_JOB)
        job_lines = [
            line for line in gate_code.splitlines()
            if len(line) - len(line.lstrip()) == 4
        ]
        job_keys = _keys("\n".join(job_lines))
        always_lines = [
            line for line in job_lines
            if re.fullmatch(r"\s*(?:['\"]?if['\"]?)\s*:\s*always\(\)\s*", line)
        ]

        self.assertEqual(
            1,
            len(always_lines),
            "gate-status must have exactly the job-level `if: always()` that reports failed needs",
        )
        self.assertNotIn(
            "continue-on-error",
            job_keys,
            "job-level continue-on-error would make the required gate-status context green",
        )
        self.assertIn("needs", job_keys, "anti-vacuity: no gate-status job-level keys were found")

        step_keys = _keys(_step_block(GATE_JOB, GATE_STEP))
        self.assertNotIn(
            "if",
            step_keys,
            "the sole gate-status reporting step became skippable, so a red census could be hidden",
        )
        self.assertNotIn(
            "continue-on-error",
            step_keys,
            "the sole gate-status reporting step may not turn a red census into a green context",
        )
        self.assertIn("run", step_keys, "anti-vacuity: no report-step keys were found")

    def test_the_job_actually_runs_the_census(self) -> None:
        step = _step_block(JOB, VERDICT_STEP)
        self.assertTrue(
            _has_line(step, INVOKE_AND_CAPTURE),
            "the verdict step no longer invokes the census AND captures its status on one line; "
            "everything else here asserts nothing. Note this is a WHOLE-LINE match: the census "
            "must be the command, not a substring of a longer pipeline whose exit code belongs to "
            "something else, and the flag list is pinned too -- `--check` was removed for being a "
            "no-op whose --help claimed a behaviour, so passing it here would be an argparse exit 2",
        )

    def test_the_verdict_is_the_exit_code_not_a_grep_of_its_own_output(self) -> None:
        """34 of `mass-gate`'s 44 steps assert on a log their own step wrote via `tee`, a shape
        held together only by `set -o pipefail` under `bash -e`. The VERDICT step must not join
        them: a success-line grep can be satisfied by an `echo`, whereas an exit code cannot.

        Scoped to that one step, not to the job. The job now also runs this module and reads its
        `Ran N tests` / `OK` tail, which is a different thing -- that output is written by a separate
        process whose status is the step's verdict -- and a job-wide `assertNotIn("tee ")` would have
        forbidden it. Scoping is also what makes the status assertion mean anything: over the whole
        job `assertIn('rc=$?', ...)` was satisfied by the KILL-CONFIRM step's own `rc=$?`, so
        deleting the one here was not caught. Found by mutation, same family as the `fetch-depth`
        comment match.

        ⚠ AND SCOPING WAS NOT ENOUGH EITHER, WHICH IS THE THIRD DISPLACEMENT OF THIS SAME PIN.
        `_has_line(step, "rc=$?")` pins that the step captures A status, not WHOSE. Review inserted
        one ordinary diagnostic `echo` between the invocation and the standalone `rc=$?` and
        measured `STEP EXIT=0` on a tree where the census exited 15 -- 22/22 pins green, both lines
        still whole lines, both arms still `exit 1`. So the capture is now part of the invocation
        (`cmd || rc=$?`), which has no second statement to displace, and a STANDALONE `rc=$?` is
        forbidden here rather than merely required to be adjacent. Pinning adjacency would have been
        the fourth version of pinning the frame instead of the property;
        `TheStepsBehaveWhenExecutedTests` is the pin that reads the property itself.
        """

        step = _step_block(JOB, VERDICT_STEP)
        self.assertNotIn(
            "tee ", step, "the verdict step now tees its output, inviting a self-grep verdict"
        )
        self.assertTrue(
            _has_line(step, INVOKE_AND_CAPTURE),
            "the census's status is no longer captured by the invocation itself, so it cannot be "
            "the verdict",
        )
        self.assertFalse(
            _has_line(step, "rc=$?"),
            "the status is captured by a STANDALONE `rc=$?` again. Any statement inserted above it "
            "orphans it silently: one `echo` there was measured at STEP EXIT=0 over a census that "
            "exited 15, with every other pin in this file green",
        )
        self.assertIn(
            "${rc}", step, "anti-vacuity: the captured status is read by nothing in this step"
        )

    def test_any_nonzero_census_exit_fails_the_step(self) -> None:
        """15 is the census's verdict and 1 is "the spec could not be read". Both must be red.

        The `!= 0` arm is the load-bearing one: without it an exit 1 -- a malformed, duplicated or
        self-contradictory spec row, or a crash -- would fall through as green, which is the
        "the gate could not run" reading the floor gate's `exit 2` arm exists to forbid.
        """

        step = _step_block(JOB, VERDICT_STEP)
        for needle in ('[ "${rc}" -eq 15 ]', '[ "${rc}" -ne 0 ]'):
            with self.subTest(arm=needle):
                self.assertEqual(
                    "exit 1",
                    _first_exit(step, needle),
                    f"the {needle} arm no longer fails the step",
                )

    def test_the_kill_confirm_demands_the_census_can_still_fail(self) -> None:
        """And that it covers each abort, not just one, and names WHICH abort it expects.

        An earlier revision perturbed only an unclassified container, which trips the PRE-EXISTING
        count-equality abort -- so the aborts added alongside this job were confirmed by nothing.
        The negative control is listed too: without it, four reddening cases prove only that
        SOMETHING reddens.

        And the exit code alone does not identify the abort: FOUR distinct aborts return 15, so an
        edit that broke case 2's KEY rather than its CLASS would still exit 15 and still pass while
        no longer exercising the per-file pin. Each case therefore also names a substring the census
        ITSELF must print. Grepping that is not the self-grep the verdict step refuses -- the log is
        written by the subject process, not by the asserting step -- and this test pins the pairing
        so a needle cannot quietly be dropped back to a bare code check.
        """

        step = _step_block(JOB, KILL_CONFIRM_STEP)
        for case, rc, needle in (
            ("unclassified container", 15, "needs its own row"),
            ("REGISTER demoted to PARTIAL", 15, "per-file classification counts do not match"),
            ("conflicting duplicate row", 1, "CONFLICTING rows for"),
            ("two containers collapsed onto one line", 15, "container node(s) collapsed into"),
            (
                "unperturbed copy (negative control)",
                0,
                "every schema-name container is classified",
            ),
        ):
            with self.subTest(case=case):
                self.assertIn(f'expect "{case}"', step, "kill-confirm case is gone")
                self.assertIn(
                    f'{rc} "{needle}"',
                    step,
                    f"case '{case}' no longer pins both its exit code and the abort text the "
                    "census must print; four aborts return 15, so the code alone does not say "
                    "which one fired",
                )
                # ...and the needle must be text the census can actually print. Without this the
                # pairing above is satisfied by a typo, and `expect` would then fail every run --
                # loudly, but for the wrong reason and only after a push.
                self.assertIn(
                    needle,
                    CENSUS.read_text(encoding="utf-8"),
                    f"the kill-confirm expects the census to print {needle!r} for case '{case}', "
                    "but no such text exists in the census",
                )
        # The verdict has to be read from the census's log, not from its code alone.
        self.assertIn(
            'grep -qF "$3"',
            step,
            "the kill-confirm no longer checks WHICH abort fired, only that something did",
        )
        self.assertEqual(
            "exit 1",
            _first_exit(step, '[ "${fail}" -ne 0 ]'),
            "a failed kill-confirm no longer fails the step, so the real run is untrusted",
        )

    def test_the_tree_censused_is_the_merge_result(self) -> None:
        """Whole-file counts must derive from the MERGE ref, not the branch.

        Both arms are required. `HEAD == head.sha` catches `ref: <head sha>`; a missing `HEAD^2`
        catches `ref: main`, which checks out a single-parent base tip. An earlier revision made
        the second a green `note:` and was fail-open for exactly that case. `fetch-depth` must
        stay >= 2 or the parent is grafted away and the second arm reddens every PR -- which it
        did, once, on this job's first CI run.
        """

        block = _job_code(JOB)
        # The KEY LINE, not a mention of it. See _job_code: the previous form of this assertion
        # was satisfied by this step's own comment, so `fetch-depth: 1` slipped through.
        self.assertTrue(
            _has_key_line(block, "fetch-depth", "2"),
            "no literal `fetch-depth: 2` key; the merge parent would be grafted away and the "
            "HEAD^2 arm would redden every PR, as it did on this job's first CI run",
        )
        # BELT, structurally, over the runtime braces above: no `ref:` on this job's checkout at
        # all. The two shell arms catch every wrong ref this job can be given, but only once a run
        # has started and only on `pull_request`; a reviewer reading the YAML should see the
        # prohibition stated where the mistake would be made. The scan skips comments, so the
        # step's own prose telling a maintainer to "remove the `ref:`" cannot satisfy it -- the
        # `fetch-depth` lesson applied before it has to be relearned.
        keys = [
            line.strip().removeprefix("- ").split(":")[0].strip().strip("\"'")
            for line in _job_code(JOB).splitlines()
            if ":" in line.strip()
        ]
        self.assertIn("fetch-depth", keys, "anti-vacuity: the checkout `with:` scan found nothing")
        self.assertNotIn(
            "ref",
            keys,
            "this job's checkout now pins a `ref:`, which is how the census comes to count a tree "
            "that is not the one that would land",
        )
        self.assertEqual(
            "exit 1",
            _first_exit(block, '[ "${head_commit}" = "${HEAD_SHA}" ]'),
            "censusing the branch tip no longer fails",
        )
        self.assertEqual(
            "exit 1",
            _first_exit(block, "if ! second_parent=$(git rev-parse --verify --quiet 'HEAD^2')"),
            "a single-parent checkout no longer fails, so `ref: main` would go green. Measured: "
            "an `echo ::notice` + `exit 0` inserted ABOVE the untouched `exit 1` was not caught "
            "while this assertion only demanded that `exit 1` appear somewhere in the arm",
        )

    def test_gate_status_needs_the_job_and_consumes_its_result(self) -> None:
        block = _job_code("gate-status")
        needs = next(line for line in block.splitlines() if line.strip().startswith("needs:"))
        self.assertIn(JOB, needs, "the census is not in the required context's needs")
        self.assertIn(f"needs.{JOB}.result", block, "the result is not read")
        self.assertIn(
            '[ "${CENSUS}" != "success" ]',
            block,
            "gate-status reads the census result but does not fail on it",
        )
        # ...and the branch must actually FAIL. `exit 1` -> `exit 0` is a one-character edit that
        # leaves every structural assertion above satisfied while the required context goes green
        # over a red census. Verified live: on the deliberate-drift commit this arm printed its
        # error and exited 1 while `mass-gate` itself was green.
        self.assertEqual(
            "exit 1",
            _first_exit(block, '[ "${CENSUS}" != "success" ]'),
            "the census branch of gate-status no longer exits nonzero",
        )

    def test_a_job_actually_runs_this_module(self) -> None:
        """THE PIN THAT MAKES EVERY OTHER PIN IN THIS FILE MEAN ANYTHING.

        At 8d32dcb2 this module executed NOWHERE. `grep -rn test_schema_container_census_wiring
        .github/ scripts/` returned zero hits, and this workflow names test modules one by one --
        35 lines carry `-m unittest`, ONE of them a comment, so 34 EXECUTABLE invocation sites
        (test_unreachable_readjudication's `python3?\\s+-m\\s*unittest` scan over the workflow,
        re-derived; the same 35/1/34 split the workflow's own note at the wiring step records, and
        an earlier form of this sentence read the 35 as invocations), no `unittest discover`, no
        pytest anywhere in .github/ -- so a new module is invisible to CI by default. Every
        assertion here was therefore local-only: a job-level `if: false`, a `continue-on-error`,
        removal from `gate-status`'s `needs:`, an `exit 1` flipped to `exit 0` or an emptied pin
        table would each have landed on main green, which is the exact defect this PR exists to
        close, reproduced one level up.

        ⚠ AND PINNING THE INVOCATION IS NOT ENOUGH, which review measured on two forges that KEEP
        the real invocation and still exit 0 over a suite that printed `FAILED (failures=1)`:

          * `| tee <file>` + `|| true`, then `printf 'Ran N tests\\nOK\\n' >> ` the same file. Both
            greps matched the appended pair. Spelled without a literal count in the forge, nothing
            else in the repository reddened either.
          * an env-gated `if [ ... ]; then <whole body> fi`. Nothing ran, no log existed, the step
            exited 0, and every assertion in this method's earlier form still held.

        So a FILE is not an acceptable subject for these guards -- any later command can extend one.
        The step captures the runner's output into a shell variable and its status on the same line,
        and the status is the verdict; `TheStepsBehaveWhenExecutedTests` executes all of that rather
        than matching it.

        ⚠ AND THE MODULE NAME AND THE CAPTURE LINE WERE TWO UNLINKED ASSERTIONS, which is a
        SUBSTITUTION seam review measured. The name check was `assertIn(<this module>, step)` --
        satisfied by ANY line of the step -- while the capture-line check tested only
        `startswith("log=$(python -m unittest tests.")` and `endswith(") || rc=$?")`, so the dotted
        name in the middle of the executed line was pinned by nothing. Swapping it for
        `tests.test_linear_policy` -- 26 tests, stdlib-only, so the count guard and the clean `OK`
        tail both still match -- ran a module with no opinion about this job at all and was MISSED
        here, caught only by `test_unreachable_readjudication` in the PATH-FILTERED `mass-gate`
        job, which does not run on a PR that touches neither. The two are now ONE assertion over
        the capture line itself, and the name is DERIVED from `__file__` rather than spelled.
        """

        step = _step_block(JOB, WIRING_STEP)
        # THE CAPTURE LINE, read once and asserted about as a whole: the module it names, the
        # variable it captures the output into, and the status capture are one property, and
        # splitting them into separate `assertIn`s is what let the name be substituted.
        captures = [
            line.strip()
            for line in step.splitlines()
            if re.match(r"^log=\$\(python3? -m ?unittest ", line.strip())
        ]
        self.assertEqual(
            [f"log=$(python -m unittest {MODULE} -v 2>&1) || rc=$?"],
            captures,
            "the step's runner line is no longer EXACTLY one invocation of THIS module whose output "
            "is captured into `log` and whose status is captured on the same line. A second such "
            "line, a different module (`tests.test_linear_policy` also has 26 stdlib-only tests, so "
            "the count guard and the OK tail keep matching), a log FILE instead of a variable, or a "
            "detached `rc=$?` each defeat this step while leaving its shape intact",
        )
        self.assertNotIn(
            "tee ",
            step,
            "the runner's output goes to a FILE again, so a later command can extend it",
        )
        self.assertNotIn(
            ">>",
            step,
            "this step appends to something; the forge that defeated its first revision was "
            "exactly an append after the runner had finished",
        )
        # Each of the three guards must exit nonzero, for the same reason gate-status's branch must:
        # `exit 1` -> `exit 0` leaves the condition, the message and the whole shape of the guard
        # intact while the step passes over a shrunk, failing or forged suite. The FIRST is the
        # load-bearing one -- the runner's status -- and the other two are shape checks on its tail.
        for needle in ('[ "${rc}" -ne 0 ]', '[ "${last}" != "OK" ]', "Ran "):
            with self.subTest(guard=needle):
                self.assertEqual(
                    "exit 1",
                    _first_exit(step, needle),
                    f"the {needle} guard no longer exits nonzero",
                )

    def test_the_workflow_test_count_guard_matches_this_module(self) -> None:
        """THE SELF-CHECK, and the property this module was missing while its precedent had it.

        `tests/test_harness_digest_provenance.py` asserts that the workflow's exact `Ran 28 tests`
        guard matches its own loader count, which is what makes forgetting that guard a LOCAL
        failure rather than a red CI run after the push. The guard lives in YAML, so no local
        `unittest` or `pytest` run can otherwise see it, and a module that grows without its count
        following is how an approved PR has reddened CI in this repo before.

        Read from a comment-stripped block, not the raw text, which is stricter than the precedent:
        a comment in the job that happened to quote the right number must not satisfy this. That is
        the same comment-matches-its-own-prose defect `_job_code` exists for.

        ⚠ AND SCOPED TO THE STEP, NOT THE JOB, which is the granularity fix `_step_block` already
        made for `rc=$?` and `_has_key_line` made for `fetch-depth`, unapplied one level up until
        review found it. `Ran 22 tests` IS NOT UNIQUE IN THIS WORKFLOW -- a second executable one
        lives in the never-fired-counter step of `mass-gate`, and this file's own step-scoped pins
        exist because a sibling's text satisfied an assertion twice already. Job-scoped, any sibling
        step of this job that came to carry the right number would satisfy this while the guard it
        is supposed to be checking was weakened to a floor.

        The pinned form is the ANCHORED one the step actually uses (`^Ran N tests in `), not a bare
        substring: the guard reads the runner's own summary line, so a form that would match text
        appended anywhere in the output must not satisfy this self-check either.
        """

        count = unittest.defaultTestLoader.loadTestsFromModule(
            sys.modules[__name__]
        ).countTestCases()
        self.assertIn(
            f"^Ran {count} tests in ",
            _step_block(JOB, WIRING_STEP),
            f"this module now has {count} tests; the `{WIRING_STEP}` step's own anchored grep guard "
            "still names a different number (or is no longer anchored to the runner's summary "
            "line), so CI would go red on a green suite",
        )
        # Anti-vacuity: a loader that found nothing would make the assertion above pass against a
        # `Ran 0 tests` guard nobody would notice was wrong.
        self.assertGreater(count, 1, "anti-vacuity: the loader found no tests in this module")


class TheStepsBehaveWhenExecutedTests(unittest.TestCase):
    """⚠ THE PIN THAT STOPS THE BOUNDARY MOVING ONE FRAME OUT AGAIN.

    Three rounds of review moved the same untested boundary four times: the call site, then
    whole-line matching, then step scoping, then ADJACENCY. Each fix pinned the frame the last
    mutation used, and the next mutation stepped one frame further out -- most recently by inserting
    a single diagnostic `echo` between the census invocation and a standalone `rc=$?`, which gave
    `STEP EXIT=0` on a census that exited 15 with all 22 pins green.

    Every one of those pins is TEXTUAL, and the property is RUNTIME. So these tests extract each
    step's own `run:` body and EXECUTE it, against stubs whose status and output they choose. A
    rearrangement of the text that breaks the relationship now fails here whatever it looks like,
    because nothing here reads the text.

    ⚠ AND ROUND 4 APPLIED THAT FIX TO TWO OF THE FIVE SHELL REGIONS THIS MODULE PINS, which is the
    round-5 finding. The verdict step and the wiring step were executed; the merge-parent step, the
    kill-confirm step and `gate-status`'s own body were left to `_first_exit`, a SIXTH textual pin
    written in the same commit -- and its `re.fullmatch(r"exit \\d+", ...)` cannot see an exit that
    is not a bare `exit N` line. Six bypasses were measured 26/26 GREEN against it, and the first
    of them made `gate-status`, the ONLY required context, exit 0 over a red census:

        exit 0 # r4        true && exit 0        exit 0;        trap 'exit 0' EXIT
        a helper function whose body exits        if false; then exit 1; fi

    The reviewer's own summary is the diagnosis, and it partitions this class exactly: *the arms
    that are EXECUTED resist; the arms that are only `_first_exit`-pinned fail open.* All five
    regions are executed here now. The remedy was never a better regex -- that would have been the
    seventh pin in a sequence whose first four were each displaced -- it was the one round 4 already
    got right, applied to the rest of its own subject.

    WHAT WOULD HAVE TO CHANGE for these pins to be satisfied while the property is false -- the
    question the textual pins never asked about themselves:

      1. **The script executed here stops being the script the runner executes.** That is the one
         real remaining frame, and it has exactly three cheap openings: a `shell:` key on either
         job or one of their steps, a workflow-level `defaults: run: shell:`, and a folded `run: >`
         whose line joining changes the script's meaning. All three are asserted against --
         `test_these_pins_execute_the_shell_the_runner_does` for the first two, `_step_script`'s
         literal-`run: |` requirement for the third.
      2. **The step reads a name its YAML `env:` block does not bind.** New with `gate-status`,
         whose whole input is seven `needs.<job>.result` bindings that `_step_script` does not
         extract: a fixture that supplies its own `CENSUS` would keep passing if the YAML renamed
         the binding, because `set -u` would only fire on the runner.
         `test_the_executed_fixtures_bind_the_names_the_steps_declare` derives the names from the
         YAML and compares them to the fixture's.
      3. **The stub is not the census.** These tests pin "the step's status tracks its subject's
         status", not "the subject's status is a correct verdict". The second is a different
         property with different pins: `TheCensusStillAbortsOnWhatItClaimsToCatchTests` below, and
         the kill-confirm step's real run in CI.
      4. So the residual is a census (or a wiring suite) that returns 0 from an abort. Structure,
         consumption and execution are all green then, and only behaviour can see it -- which is why
         (3) is a separate battery rather than an extension of this one.
    """

    #: Every gate `gate-status` reports on, at its green value. Fixtures below copy this and spoil
    #: ONE entry, so a red verdict is attributable to that entry and not to the fixture's shape.
    #: `set -u` is on in that step, so an incomplete dict here would redden every case for the
    #: wrong reason -- which is what the all-green control catches.
    GREEN_GATE_ENV = {
        "TOUCHED": "true",
        "RESULT": "success",
        "FILTER": "success",
        "REGISTRY": "success",
        "HARNESS": "success",
        "FLOOR": "success",
        "CENSUS": "success",
    }

    #: The kill-confirm step's five scratch case directories, each with the exit code and the abort
    #: text that case's `expect()` demands. Keyed on the directory name because that is what the
    #: step passes the census, and therefore the only thing a stub can dispatch on.
    KILL_CONFIRM_VERDICTS = {
        "unclassified": (15, "needs its own row"),
        "classflip": (15, "per-file classification counts do not match"),
        "conflict": (1, "CONFLICTING rows for"),
        "collapse": (15, "container node(s) collapsed into"),
        "clean": (0, "every schema-name container is classified"),
    }

    def _execute(
        self,
        step_name: str,
        *,
        status: int | None = None,
        stub_output: str = "",
        job: str = JOB,
        env: dict[str, str] | None = None,
        cwd: pathlib.Path | None = None,
        stub: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one step's `run:` body under the runner's own shell, with `python` stubbed.

        The default stub writes `stub_output` to STDERR because that is where `unittest` writes,
        and exits `status`. It never runs the real census or the real suite: the wiring step
        invokes THIS module, so a passthrough stub would recurse.

        `stub` replaces that body outright, for the kill-confirm step, whose five cases each need
        their own verdict from one `python` on the PATH. `env` and `cwd` exist for the two regions
        whose input is not a stub at all: `gate-status` reads seven `needs.*.result` names under
        `set -u`, and the merge-parent step reads a git history rather than a command's status.
        """

        script = _step_script(job, step_name)
        root = pathlib.Path(tempfile.mkdtemp(prefix="census-step-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "bin").mkdir()
        (root / "runner-temp").mkdir()
        if stub is None:
            stub = '#!/bin/sh\nprintf "%s" "${STUB_OUTPUT}" >&2\n' f"exit {status}\n"
        stub_path = root / "bin" / "python"
        stub_path.write_text(stub, encoding="utf-8")
        stub_path.chmod(0o755)
        (root / "step.sh").write_text(script, encoding="utf-8")
        environ = dict(os.environ)
        environ["PATH"] = f"{root / 'bin'}{os.pathsep}{environ['PATH']}"
        environ["STUB_OUTPUT"] = stub_output
        # The runner sets this and the kill-confirm step builds its scratch trees under it. A
        # fixture that let it default to the real runner's value would write outside the tmpdir
        # this test cleans up.
        environ["RUNNER_TEMP"] = str(root / "runner-temp")
        environ.update(env or {})
        # `bash -e` with no `-o pipefail`, which is what GitHub hands a `run:` body that names no
        # `shell:`. Steps that want pipefail say so themselves, and every step here does.
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", str(root / "step.sh")],
            cwd=str(cwd or REPO),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            env=environ,
        )

    def _merge_repo(self) -> tuple[pathlib.Path, str, str, str]:
        """A throwaway git repo holding the shape `actions/checkout@v4` leaves on `pull_request`.

        Returns the root and the three commits the merge-parent step can be pointed at: the base
        tip, the PR head, and the merge of the two, which is what `refs/pull/N/merge` resolves to
        and the only tree whose whole-file counts are the ones that would land.

        Identity comes from the environment rather than `git config`, and both config files are
        pointed at /dev/null, so a developer's global `commit.gpgsign` or hook path cannot make
        this fixture fail for a reason that has nothing to do with its subject.
        """

        root = pathlib.Path(tempfile.mkdtemp(prefix="census-merge-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        environ = dict(os.environ)
        environ.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_SYSTEM=os.devnull,
            GIT_AUTHOR_NAME="census fixture",
            GIT_AUTHOR_EMAIL="census@example.invalid",
            GIT_COMMITTER_NAME="census fixture",
            GIT_COMMITTER_EMAIL="census@example.invalid",
        )

        def git(*args: str) -> str:
            done = subprocess.run(
                ["git", *args], cwd=str(root), capture_output=True, text=True,
                timeout=_TIMEOUT, env=environ,
            )
            self.assertEqual(done.returncode, 0, f"git {args}: {done.stdout}{done.stderr}")
            return done.stdout.strip()

        git("init", "--quiet")
        # Not `init --initial-branch`, which predates neither git 2.28 nor every runner image.
        git("symbolic-ref", "HEAD", "refs/heads/base")
        git("commit", "--quiet", "--allow-empty", "-m", "base")
        base = git("rev-parse", "HEAD")
        git("checkout", "--quiet", "-b", "feature")
        git("commit", "--quiet", "--allow-empty", "-m", "the pull request")
        head = git("rev-parse", "HEAD")
        git("checkout", "--quiet", "base")
        git("merge", "--quiet", "--no-ff", "-m", "merge", "feature")
        merge = git("rev-parse", "HEAD")
        self.assertEqual(head, git("rev-parse", "HEAD^2"), "fixture: HEAD^2 is not the PR head")
        return root, base, head, merge

    def _detach(self, root: pathlib.Path, commit: str) -> None:
        """Point the fixture repo's HEAD at one commit, as a `ref:` on the checkout would."""

        done = subprocess.run(
            ["git", "checkout", "--quiet", "--detach", commit],
            cwd=str(root), capture_output=True, text=True, timeout=_TIMEOUT,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def _kill_confirm_tree(self) -> pathlib.Path:
        """The smallest tree the kill-confirm step's own commands run against.

        Minimal rather than the repo, for the reason the behaviour battery is: `cp -R src scripts`
        runs once per case, and a fixture whose answer moves when a file is added to src/ is not a
        fixture. The REAL spec file is copied in, though, because two of the five cases perturb it
        with commands that must find something -- case 2's `perl` needs the `REGISTER
        src/pokezero/rollout_cli.py` row and case 3's `grep -m1 '^REGISTER'` fails the step outright
        under `set -o pipefail` if nothing matches. Only the interpreter is stubbed.
        """

        root = pathlib.Path(tempfile.mkdtemp(prefix="census-killconfirm-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "src" / "pokezero").mkdir(parents=True)
        (root / "src" / "pokezero" / "__init__.py").write_text("", encoding="utf-8")
        (root / "scripts").mkdir()
        shutil.copy(CENSUS, root / "scripts" / CENSUS.name)
        (root / "tests" / "data").mkdir(parents=True)
        shutil.copy(SPEC, root / "tests" / "data" / SPEC.name)
        return root

    def _kill_confirm_stub(self, verdicts: dict[str, tuple[int, str]]) -> str:
        """A `python` that answers per SCRATCH CASE, so one case can be spoiled and four behave.

        Anchored on `census-kill-confirm/<case>/` rather than on `<case>` alone: the enclosing
        tmpdir name is random, and a glob loose enough to match it somewhere else would make a
        spoiled case pass by accident.
        """

        arms = "\n".join(
            f"  */census-kill-confirm/{case}/*)\n"
            f"    printf '%s\\n' {shlex.quote(text)}\n"
            f"    exit {rc} ;;"
            for case, (rc, text) in verdicts.items()
        )
        return (
            "#!/bin/sh\n"
            "# Stands in for the COPIED census. Its exit code and its abort text are chosen by\n"
            "# which scratch case invoked it; the real one is never run here.\n"
            'case "$1" in\n'
            f"{arms}\n"
            "  *)\n"
            "    printf '%s\\n' \"stub reached for an unrecognized case: $1\"\n"
            "    exit 99 ;;\n"
            "esac\n"
        )

    def test_the_verdict_steps_status_is_the_censuss_status(self) -> None:
        """THE B1 PIN, stated as the relationship rather than as a shape.

        The failing input this closes: one `echo "census finished"` inserted between the invocation
        and `rc=$?` printed `census exit code: 0` and exited 0 while the census exited 15.
        """

        # 0 is the CONTROL: without it, a step body that had stopped extracting (or that failed for
        # an unrelated reason) would satisfy every red case below and prove nothing.
        self.assertEqual(
            self._execute(VERDICT_STEP, status=0).returncode,
            0,
            "control: a census that exits 0 must leave the verdict step green",
        )
        for status in (15, 1, 2, 127):
            with self.subTest(census_exit=status):
                result = self._execute(VERDICT_STEP, status=status)
                self.assertEqual(
                    result.returncode,
                    1,
                    f"the census exited {status} and the verdict step exited "
                    f"{result.returncode}. The step's status no longer tracks the census's -- which "
                    "is what an orphaned `rc=$?` does, silently.\n" + result.stdout + result.stderr,
                )

    def test_the_wiring_steps_verdict_is_the_runners_status(self) -> None:
        """THE B2 PIN. Both measured forges keep the invocation and forge only the OUTPUT, so the
        status has to be the verdict. Here the stub's output is a PERFECT clean tail in every case
        and only the status varies: a step that reads its log instead of its runner passes all of
        these, and must not.
        """

        count = unittest.defaultTestLoader.loadTestsFromModule(
            sys.modules[__name__]
        ).countTestCases()
        clean = f"Ran {count} tests in 0.012s\n\nOK\n"
        self.assertEqual(
            self._execute(WIRING_STEP, status=0, stub_output=clean).returncode,
            0,
            "control: a clean run must leave the wiring step green",
        )
        for status in (1, 2, 127):
            with self.subTest(runner_exit=status):
                result = self._execute(WIRING_STEP, status=status, stub_output=clean)
                self.assertEqual(
                    result.returncode,
                    1,
                    f"the runner exited {status} behind a clean-looking tail and the step exited "
                    f"{result.returncode}. This is the `|| true` + appended-summary forge: it was "
                    "measured landing green over a suite whose own log said FAILED.\n"
                    + result.stdout
                    + result.stderr,
                )

    def test_the_wiring_step_rejects_a_tail_the_runner_did_not_write(self) -> None:
        """The other half of B2: with the status forced to 0, the OUTPUT must still be the runner's.

        The empty case is the env-gated no-op wrapper -- nothing runs, nothing is written, and the
        first revision of this step exited 0. The trailing-`FAILED` case is what an append is trying
        to bury. The wrong-count case is the exact-count guard, which is what the workflow-side
        self-check keeps honest.
        """

        count = unittest.defaultTestLoader.loadTestsFromModule(
            sys.modules[__name__]
        ).countTestCases()
        for case, output in (
            ("nothing ran at all", ""),
            ("no summary line", "OK\n"),
            ("wrong count", f"Ran {count - 1} tests in 0.012s\n\nOK\n"),
            ("summary present but the run failed", f"Ran {count} tests in 0.012s\n\nFAILED (failures=1)\n"),
            (
                "clean tail buried above appended noise",
                f"Ran {count} tests in 0.012s\n\nOK\nand then something else\n",
            ),
        ):
            with self.subTest(case=case):
                result = self._execute(WIRING_STEP, status=0, stub_output=output)
                self.assertEqual(
                    result.returncode,
                    1,
                    f"the wiring step accepted '{case}' as a pass\n" + result.stdout + result.stderr,
                )

    def test_the_merge_parent_step_reddens_on_every_wrong_checkout(self) -> None:
        """REGION 3 OF 5, and the first of the three round 4 left to `_first_exit`.

        Two of this step's three arms were pinned by a first-exit assertion and the third
        (`HEAD^2 != HEAD_SHA`) by NOTHING AT ALL. Measured against the textual pin: `exit 0 # r4`,
        `true && exit 0`, `exit 0;`, `trap 'exit 0' EXIT`, a helper function that exits, and
        `if false; then exit 1; fi` were each 26/26 green in either arm. Executed, each of the
        four checkouts below decides the step's status, so a mutation that stops an arm from
        failing is red here regardless of how it spells not exiting.

        The step reads a git HISTORY rather than a command's status, so the fixture is a repo and
        not a stub: three commits in the shape `actions/checkout@v4` leaves on `pull_request`, and
        HEAD pointed at each of the wrong ones in turn.

        ⚠ AND THE EXIT CODE ALONE DOES NOT IDENTIFY WHICH ARM FIRED, which the first revision of
        this test got wrong and its own battery caught: THE THREE ARMS MASK EACH OTHER. Neuter arm
        1 and the branch-tip checkout falls through to arm 2, because a branch tip has no `HEAD^2`
        either -- exit 1, test green, arm dead. Neuter arm 2 and the failed command substitution
        leaves `second_parent` set to the empty string, so arm 3 fires on `"" != HEAD_SHA` -- exit
        1, test green, arm dead. Both were live MISSES with `if false; then exit 1; fi`.

        ⚠ AND NAMING THE ARM'S OWN `::error::` LINE WAS NOT ENOUGH EITHER, which is the same
        mistake one frame further in: the echoes come BEFORE the exit, so a neutered arm still
        prints its message and the needle was present on both misses. What distinguishes "this arm
        decided" from "this arm ran and something later decided" is that NO OTHER ARM SPOKE. So
        each case asserts the set of arm messages in the log is EXACTLY its own -- which also makes
        the case that reddens via a sibling report the sibling by name.
        """

        script = _step_script(JOB, MERGE_STEP)
        # Each arm's own line, and it must identify that arm ALONE: these are compared as a set
        # below, so a needle shared by two arms would make the comparison unable to tell them
        # apart -- the `_has_line` lesson, in the observations rather than in the assertions.
        arms = {
            "the BRANCH TIP arm": "so the census is counting the BRANCH TIP",
            "the missing-HEAD^2 arm": "HEAD has no second parent",
            "the wrong-HEAD^2 arm": "censused is not the merge of this pull request",
        }
        for arm, needle in arms.items():
            self.assertEqual(
                1, script.count(needle),
                f"anti-vacuity: {needle!r} does not appear exactly once in this step, so it cannot "
                f"identify {arm}",
            )
        confirmed = "confirmed: HEAD is the merge of the base with PR head"
        self.assertIn(confirmed, script, "anti-vacuity: the step has no success line to look for")

        def spoke(log: str) -> set[str]:
            return {arm for arm, needle in arms.items() if needle in log}

        root, base, head, merge = self._merge_repo()
        # THE CONTROL, and it is load-bearing three times over: it says the fixture repo is the
        # shape the step expects, it says the red cases below are red because of their arm rather
        # than because `set -u` tripped over a name this fixture forgot to bind, and its own needle
        # says the step reached its end rather than exiting 0 without doing anything.
        clean = self._execute(MERGE_STEP, env={"HEAD_SHA": head}, cwd=root)
        self.assertEqual(
            clean.returncode, 0,
            "control: censusing the merge of the base with the PR head must be green\n"
            + clean.stdout + clean.stderr,
        )
        self.assertEqual(
            (set(), True),
            (spoke(clean.stdout), f"{confirmed} {head}" in clean.stdout),
            "control: the merge result must reach the step's own confirmation with no arm speaking\n"
            + clean.stdout + clean.stderr,
        )
        for case, at, head_sha, arm in (
            ("`ref: <head sha>` -- the BRANCH TIP", head, head, "the BRANCH TIP arm"),
            ("`ref: main` -- a single-parent BASE TIP", base, head, "the missing-HEAD^2 arm"),
            ("the merge of a DIFFERENT pull request", merge, base, "the wrong-HEAD^2 arm"),
        ):
            with self.subTest(checkout=case):
                self._detach(root, at)
                result = self._execute(MERGE_STEP, env={"HEAD_SHA": head_sha}, cwd=root)
                self.assertEqual(
                    result.returncode,
                    1,
                    f"the census would have counted {case} and the step exited "
                    f"{result.returncode}; {arm} no longer fails the step, so whole-file counts "
                    "would derive from a tree that is not the one that would land.\n"
                    + result.stdout
                    + result.stderr,
                )
                # WHICH ARM DECIDED. Exactly its own, and the step must not have reached the end.
                # A neutered arm still prints its message and then falls through, so "my needle is
                # present" is satisfied by a dead arm; "no other arm spoke" is not.
                self.assertEqual(
                    {arm},
                    spoke(result.stdout),
                    f"the step reddened on {case}, but {arm} is not what decided: the arms that "
                    f"spoke were {sorted(spoke(result.stdout))}. The three mask each other -- a "
                    "branch tip has no HEAD^2 either, and a failed `HEAD^2` leaves `second_parent` "
                    "empty for the arm after it -- so an arm can print its error, fall through, "
                    "and be red on its successor's verdict.\n"
                    + result.stdout
                    + result.stderr,
                )
                self.assertNotIn(
                    confirmed,
                    result.stdout,
                    f"the step printed its CONFIRMATION on {case} and reddened anyway",
                )

    def test_the_kill_confirm_step_reddens_when_a_case_misses_its_verdict(self) -> None:
        """REGION 4 OF 5. The `fail` arm is what makes four aborts and a control into a gate.

        Its `exit 1` was `_first_exit`-pinned only, and the same six bypasses applied to it were
        26/26 green -- so the step could run all five cases, print `::error::` for a case that
        returned the wrong code, and exit 0. That is worse than no kill-confirm: the real census
        run in the next step is trusted BECAUSE this one passed.

        Each case is spoiled ONE AT A TIME and in two independent ways, because the step has two
        rejection arms and a fixture that spoils both at once pins neither -- the code check
        short-circuits the needle check. `expect` is called five times per execution, so a spoiled
        case must also be shown not to be masked by the four that behave.
        """

        tree = self._kill_confirm_tree()
        verdicts = self.KILL_CONFIRM_VERDICTS
        # THE CONTROL: every case answering exactly what its `expect` demands. Without it a step
        # body that had stopped extracting, or a stub the step never reached, would satisfy every
        # red case below and prove nothing.
        clean = self._execute(
            KILL_CONFIRM_STEP, cwd=tree, stub=self._kill_confirm_stub(verdicts)
        )
        self.assertEqual(
            clean.returncode,
            0,
            "control: five cases each returning the code and the abort text they name must leave "
            "the kill-confirm green\n" + clean.stdout + clean.stderr,
        )
        for case, (rc, text) in verdicts.items():
            for how, spoiled in (
                # A DIFFERENT EXIT CODE. 15 and 0 are each other's wrong answer here, which also
                # covers the negative control's direction: a `clean` copy that aborts is as much a
                # broken kill-confirm as a perturbed copy that passes.
                ("returns the wrong exit code", (0 if rc else 15, text)),
                # THE RIGHT CODE VIA THE WRONG ABORT -- the case the needle arm exists for, and the
                # one a bare code check cannot see, because four distinct aborts return 15.
                ("returns the right code but never prints its abort", (rc, "an unrelated abort")),
            ):
                with self.subTest(case=case, spoiled=how):
                    result = self._execute(
                        KILL_CONFIRM_STEP,
                        cwd=tree,
                        stub=self._kill_confirm_stub({**verdicts, case: spoiled}),
                    )
                    self.assertEqual(
                        result.returncode,
                        1,
                        f"kill-confirm case '{case}' {how} and the step exited "
                        f"{result.returncode}. The `fail` arm no longer fails the step, so the "
                        "census's real run below is trusted on the strength of a kill-confirm that "
                        "reported its own failure and passed anyway.\n"
                        + result.stdout
                        + result.stderr,
                    )

    def test_gate_status_actually_exits_nonzero_over_a_red_census(self) -> None:
        """REGION 5 OF 5, AND THE ONE THAT MATTERS MOST: `gate-status` is the only required context.

        Every other pin in this file is about a job whose result nothing has to read. This is the
        arm that makes the census non-advisory, and it was `_first_exit`-pinned only. Measured with
        `CENSUS=failure`: all six bypasses -- including `if false; then exit 1; fi`, which adds no
        `exit 0` for any regex to find -- exited 0 over a red census with 26/26 green. `gate-status`
        was green on cb310479 itself, produced by the one arm nothing executed.

        `skipped` and the empty string are in the sweep because the arm demands the POSITIVE rather
        than excluding a list of negatives, and that choice is only worth anything if the negatives
        it was not written against are red too. The empty string is what a job that never ran
        leaves behind.
        """

        control = self._execute(GATE_STEP, job=GATE_JOB, env=dict(self.GREEN_GATE_ENV))
        self.assertEqual(
            control.returncode,
            0,
            "control: every gate green must leave the required context green -- otherwise the red "
            "cases below prove only that this fixture is missing an env name `set -u` needs\n"
            + control.stdout
            + control.stderr,
        )
        for result_value in ("failure", "cancelled", "skipped", ""):
            with self.subTest(census=result_value or "<the empty string>"):
                run = self._execute(
                    GATE_STEP,
                    job=GATE_JOB,
                    env={**self.GREEN_GATE_ENV, "CENSUS": result_value},
                )
                self.assertEqual(
                    run.returncode,
                    1,
                    f"the schema-container census result was {result_value!r} and the REQUIRED "
                    f"context exited {run.returncode}. The census is advisory again: it runs, it "
                    "aborts, and branch protection is satisfied by a green gate-status that never "
                    "looked at it.\n" + run.stdout + run.stderr,
                )
                # ...and via the CENSUS arm, not via a sibling. Every other gate is green in this
                # fixture so no sibling can fire today, but the merge-parent step's three arms were
                # each found masking the next, and this step has seven in a row.
                self.assertIn(
                    "the schema-container census did not pass",
                    run.stdout,
                    "gate-status reddened, but not via its census arm\n" + run.stdout + run.stderr,
                )

    def test_gate_status_exits_nonzero_over_every_other_gate_it_reports(self) -> None:
        """The census arm above must not be the only one executed, for the reason `_step_block`
        exists: an assertion that holds for one arm says nothing about its four siblings, and all
        five were written from the same template and carry the same failure mode. Two of them
        (`mass-gate`'s skipped-while-touched arm, and the path filter's own verdict) are the arms
        that once printed "gates correctly not run" and exited 0 on a PR whose gates never ran.
        """

        for case, env in (
            ("seed-registry red", {"REGISTRY": "failure"}),
            ("harness-provenance red", {"HARNESS": "failure"}),
            ("python-floor-syntax red", {"FLOOR": "failure"}),
            ("the path filter itself red", {"FILTER": "failure"}),
            ("the path filter produced no verdict", {"TOUCHED": ""}),
            ("mass-gate red", {"RESULT": "failure"}),
            ("mass-gate cancelled", {"RESULT": "cancelled"}),
            ("mass-gate skipped while fidelity paths WERE touched",
             {"RESULT": "skipped", "TOUCHED": "true"}),
        ):
            with self.subTest(case=case):
                run = self._execute(
                    GATE_STEP, job=GATE_JOB, env={**self.GREEN_GATE_ENV, **env}
                )
                self.assertEqual(
                    run.returncode,
                    1,
                    f"'{case}' left the required context exiting {run.returncode}\n"
                    + run.stdout
                    + run.stderr,
                )
        # ...and the one skip that IS legitimate must stay green, or the arms above are satisfied by
        # a step that fails on everything and the required context can never pass.
        allowed = self._execute(
            GATE_STEP,
            job=GATE_JOB,
            env={**self.GREEN_GATE_ENV, "RESULT": "skipped", "TOUCHED": "false"},
        )
        self.assertEqual(
            allowed.returncode,
            0,
            "control: mass-gate skipped with no fidelity paths touched is the one deliberate skip\n"
            + allowed.stdout
            + allowed.stderr,
        )

    def test_the_executed_fixtures_bind_the_names_the_steps_declare(self) -> None:
        """Frame 2 from the class docstring: `_step_script` extracts the `run:` body and nothing else.

        So the env a step reads is a SECOND hand-maintained agreement between this module and the
        YAML, invisible to every other pin here. Rename `CENSUS` to `CENSUS_RESULT` in
        `gate-status`'s `env:` block and the executed pins above keep passing on their own fixture
        while the real step dies on `set -u` -- red in CI, but only after a push, and for a reason
        the local suite cannot state. Derived from the YAML and compared both ways, so a name added
        there without a fixture is as loud as a fixture name the YAML dropped.
        """

        self.assertEqual(
            set(self.GREEN_GATE_ENV),
            _step_env_names(GATE_JOB, GATE_STEP),
            f"the `{GATE_STEP}` step's `env:` block and this class's GREEN_GATE_ENV no longer bind "
            "the same names, so the executed gate-status pins are running against a fixture the "
            "runner would not reproduce",
        )
        self.assertEqual(
            {"HEAD_SHA"},
            _step_env_names(JOB, MERGE_STEP),
            f"the `{MERGE_STEP}` step's `env:` block changed; its executed pin binds HEAD_SHA only",
        )
        # And the census result must still be plumbed from the job this file is about, not from a
        # name that merely looks like it. This is the value half of the agreement above.
        self.assertIn(
            f"CENSUS: ${{{{ needs.{JOB}.result }}}}",
            _step_block(GATE_JOB, GATE_STEP),
            "gate-status's CENSUS no longer reads this job's result",
        )

    def test_these_pins_execute_the_shell_the_runner_does(self) -> None:
        """Frame 1 from the class docstring, closed rather than recorded.

        The tests above execute the text this module extracts. If the runner interprets that text
        differently they measure nothing -- and `shell:` on a job or step, and a workflow-level
        `defaults: run: shell:`, are the two ways to arrange that in this file. Neither exists, and
        neither may: the default for a `run:` body on `ubuntu-latest` is `bash -e {0}`, which is
        what `_execute` invokes.

        BOTH JOBS, not just the census job: `gate-status`'s own `run:` body is executed here now,
        so its interpreter is load-bearing too and was outside this pin's scope while only the
        census job was executed.

        ⚠ AND THE `defaults:` HALF WAS A RAW SUBSTRING (`assertNotIn("\\ndefaults:", text)`), which
        `defaults :` -- a space before the colon, valid YAML, the very variant this file already
        normalizes against for `if :` one class up -- walked past. Routed through `_keys` now, which
        also covers `"defaults":` and `'defaults' :`. Review could not get that spelling to fail
        OPEN under `sh -e` or with `-e` dropped, so it was a latent seam rather than a live
        bypass; it is closed at the cost of one helper call because the next spelling might not be.
        """

        for job in (JOB, GATE_JOB):
            with self.subTest(job=job):
                self.assertNotIn(
                    "shell",
                    _keys(_job_block(job)),
                    f"the {job} job now names a `shell:`, so the executed pins above may be running "
                    "a different interpreter than CI does",
                )
        self.assertNotIn(
            "defaults",
            _top_level_keys(),
            "this workflow now sets top-level `defaults:`, which can change the shell every `run:` "
            "body is interpreted by without touching either job at all",
        )
        # Anti-vacuity for BOTH scans: a key scan that found nothing would satisfy every
        # prohibition above. `on` and `jobs` are the two keys a workflow cannot omit.
        self.assertIn("jobs", _top_level_keys(), "anti-vacuity: the top-level key scan found nothing")
        self.assertIn("runs-on", _keys(_job_block(GATE_JOB)), f"anti-vacuity: no keys in {GATE_JOB}")
        # ...and the extraction must actually be producing the scripts those pins run.
        for job, step_name, token in (
            (JOB, VERDICT_STEP, "python"),
            (JOB, WIRING_STEP, "python"),
            (JOB, MERGE_STEP, "git rev-parse"),
            (JOB, KILL_CONFIRM_STEP, "python"),
            (GATE_JOB, GATE_STEP, "${CENSUS}"),
        ):
            with self.subTest(step=step_name):
                self.assertIn(token, _step_script(job, step_name))


class TheCensusStillAbortsOnWhatItClaimsToCatchTests(unittest.TestCase):
    """Structure and consumption are both satisfied by a census that returns 0 unconditionally."""

    def _tree(self, sources: dict[str, str], rows: str, pin: dict[str, dict[str, int]]) -> pathlib.Path:
        """A minimal repo: the census, one or more src files, and a spec.

        `scripts/` holds only the census, which excludes itself by resolved path, so it
        contributes no containers and every count below comes from `sources` alone.
        """

        root = pathlib.Path(tempfile.mkdtemp(prefix="census-wiring-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "scripts").mkdir()
        (root / "src").mkdir()
        (root / "tests" / "data").mkdir(parents=True)

        text = CENSUS.read_text(encoding="utf-8")
        # Substitute the pin, and ASSERT the substitution landed: if a reformat breaks this regex,
        # these fixtures would silently start testing the real repo's pin against a synthetic
        # tree, which would fail for the wrong reason and teach the reader nothing.
        text, n = re.subn(
            rf"{_PIN_NAME} = \{{.*?\n\}}",
            f"{_PIN_NAME} = {pin!r}",
            text,
            flags=re.DOTALL,
        )
        self.assertEqual(n, 1, f"could not substitute {_PIN_NAME}; has its literal been reformatted?")
        (root / "scripts" / "schema_container_census.py").write_text(text, encoding="utf-8")

        for name, body in sources.items():
            (root / "src" / name).write_text(body, encoding="utf-8")
        (root / "tests" / "data" / "schema_drill_schema_containers.txt").write_text(
            rows, encoding="utf-8"
        )
        return root

    def _run(self, root: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(root / "scripts" / "schema_container_census.py"), *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )

    ONE = {"one.py": 'TABLE = {"v3", "v4"}\n'}
    ONE_ROW = "REGISTER  src/one.py  v3,v4  TABLE\n"
    ONE_PIN = {"src/one.py": {"REGISTER": 1}}

    def test_a_classified_tree_is_green(self) -> None:
        """Anti-vacuity for every red fixture below: this exact shape must pass."""

        result = self._run(self._tree(self.ONE, self.ONE_ROW, self.ONE_PIN))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_an_unclassified_container_aborts(self) -> None:
        root = self._tree(
            {**self.ONE, "two.py": 'OTHER = {"v2", "v2.1"}\n'}, self.ONE_ROW, self.ONE_PIN
        )
        self.assertEqual(self._run(root).returncode, 15)

    def test_a_stale_row_aborts(self) -> None:
        root = self._tree(
            self.ONE,
            self.ONE_ROW + "REGISTER  src/gone.py  v3,v4  VANISHED\n",
            self.ONE_PIN,
        )
        self.assertEqual(self._run(root).returncode, 15)

    def test_a_class_flip_aborts_even_though_the_row_still_exists(self) -> None:
        """THE HOLE THIS PIN CLOSES. Counts stay equal, every container has a row, and the class
        is the only thing that moved -- so every other check in the census passes. Before the pin
        this was exit 0 for one row and for all ten, while the drill's registration list shrank.
        """

        root = self._tree(self.ONE, "PARTIAL   src/one.py  v3,v4  TABLE\n", self.ONE_PIN)
        result = self._run(root)
        self.assertEqual(result.returncode, 15, result.stdout)
        self.assertIn("per-file classification counts", result.stdout)

    def test_a_class_permutation_across_files_aborts(self) -> None:
        """Pinning three GLOBAL totals left every permutation free; REGISTER<->MIRRORED was the
        dangerous one, being the subtle judgement between the two large classes. Per-file counts
        are what make a swap visible, because it changes two entries rather than none.
        """

        sources = {"one.py": 'TABLE = {"v3", "v4"}\n', "two.py": 'OTHER = {"v2", "v2.1"}\n'}
        pin = {"src/one.py": {"REGISTER": 1}, "src/two.py": {"MIRRORED": 1}}
        swapped = "MIRRORED  src/one.py  v3,v4  TABLE\nREGISTER  src/two.py  v2,v2.1  OTHER\n"
        result = self._run(self._tree(sources, swapped, pin))
        self.assertEqual(result.returncode, 15, result.stdout)
        # Anti-vacuity: the unswapped spec, same tree and same pin, must be green -- otherwise
        # this fixture could be passing on an unrelated abort.
        straight = "REGISTER  src/one.py  v3,v4  TABLE\nMIRRORED  src/two.py  v2,v2.1  OTHER\n"
        self.assertEqual(self._run(self._tree(sources, straight, pin)).returncode, 0)

    def test_conflicting_rows_for_one_container_abort(self) -> None:
        """Exit 1, not 15: the spec cannot be read, rather than the tree being wrong.

        The arithmetic is the point -- `classification()` builds a dict, so a duplicate key used
        to SHRINK `len(spec)`, which is the one direction the count-equality check cannot see. Two
        rows against one container read as "1 container / 1 row" and went green.
        """

        root = self._tree(
            self.ONE, self.ONE_ROW + "PARTIAL   src/one.py  v3,v4  TABLE\n", self.ONE_PIN
        )
        result = self._run(root)
        self.assertEqual(result.returncode, 1)
        self.assertIn("CONFLICTING", result.stdout + result.stderr)

    def test_two_unnamed_containers_on_one_line_abort(self) -> None:
        """Used to print "1 container / 1 row" and exit 0 on a file holding two."""

        root = self._tree(
            {"one.py": 'def f(x):\n    return x in {"v3", "v4"} or x in {"v3", "v4"}\n'},
            "PARTIAL   src/one.py  v3,v4  <inline:2>\n",
            {"src/one.py": {"PARTIAL": 1}},
        )
        result = self._run(root)
        self.assertEqual(result.returncode, 15, result.stdout)
        self.assertIn("collapsed", result.stdout)

    def test_the_same_containers_on_separate_lines_are_green(self) -> None:
        """Isolates the collapse abort: it must key on ONE LINE, not on repeated members."""

        root = self._tree(
            {"one.py": 'def f(x):\n    a = x in {"v3", "v4"}\n    b = x in {"v3", "v4"}\n'
                       '    return a or b\n'},
            "PARTIAL   src/one.py  v3,v4  <inline:2>\nPARTIAL   src/one.py  v3,v4  <inline:3>\n",
            {"src/one.py": {"PARTIAL": 2}},
        )
        result = self._run(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_every_abort_applies_whatever_the_flags(self) -> None:
        """`--list` used to print the stale rows and the unclassified containers of the exact
        e496e9b8 shape and then exit 0 -- a diagnostic that shows a contradiction and reports
        success, which is the same defect as a gate that states one and ignores it.

        `--list` is now the ONLY flag. The `--check` this test used to be named for was a complete
        no-op: `args.check` was read nowhere, the two outputs were byte-identical, and its `--help`
        described what `--list` does. It is removed, so the argument surface is one flag and the
        list of flags to sweep here is exhaustive rather than a sample.
        """

        root = self._tree(
            {**self.ONE, "two.py": 'OTHER = {"v2", "v2.1"}\n'}, self.ONE_ROW, self.ONE_PIN
        )
        for args in (("--list",), ()):
            with self.subTest(args=args):
                self.assertEqual(self._run(root, *args).returncode, 15)

    def test_the_removed_check_flag_is_rejected_rather_than_silently_accepted(self) -> None:
        """And that the sweep above really is exhaustive.

        A no-op flag is worse than a missing one, because `--help` then has to describe it and the
        description was false. This asserts the removal is real -- argparse exit 2 with the flag
        named -- and, so the claim in `--help` cannot come back, that no help text mentions it.
        """

        root = self._tree(self.ONE, self.ONE_ROW, self.ONE_PIN)
        stale = self._run(root, "--check")
        self.assertEqual(stale.returncode, 2, stale.stdout + stale.stderr)
        self.assertIn("--check", stale.stderr, "argparse did not name the unrecognized flag")
        helped = self._run(root, "--help")
        self.assertEqual(helped.returncode, 0, helped.stderr)
        self.assertNotIn(
            "--check",
            helped.stdout,
            "`--help` still advertises `--check`; its text claimed the flag widens the listing, "
            "which is what `--list` does, and `args.check` was read nowhere",
        )
        self.assertIn("--list", helped.stdout, "anti-vacuity: no flags in --help at all")


class TheCommittedSpecMatchesTheCommittedPinTests(unittest.TestCase):
    """The repo's own tree, so a drift on main is a test failure and not only a CI failure."""

    def test_the_census_passes_on_this_tree(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CENSUS)],
            cwd=str(REPO), capture_output=True, text=True, timeout=_TIMEOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_the_pin_is_derived_from_the_spec_and_not_merely_present(self) -> None:
        """Anti-vacuity for the pin itself: an empty or stale pin table would make
        `EXPECTED_CLASS_COUNTS_BY_FILE` decorative. Recomputed here from the spec file directly.
        """

        # Parsed as a LITERAL, not exec'd. A line-scraping `exec` was the first form of this test
        # and it swept up any docstring line that happened to start with four spaces and a quote,
        # then died on the indent -- a test that fails for a reason unrelated to its subject.
        match = re.search(rf"^{_PIN_NAME} = (\{{.*?\n\}})", CENSUS.read_text(encoding="utf-8"),
                          flags=re.DOTALL | re.MULTILINE)
        self.assertIsNotNone(match, f"could not find the {_PIN_NAME} literal")
        pinned = ast.literal_eval(match.group(1))

        expected: dict[str, dict[str, int]] = {}
        for raw in SPEC.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            kind, path = line.split()[0], line.split()[1]
            expected.setdefault(path, {})
            expected[path][kind] = expected[path].get(kind, 0) + 1

        self.assertEqual(pinned, expected, "the pin no longer agrees with the spec file")
        self.assertTrue(expected, "anti-vacuity: no rows parsed out of the spec")


if __name__ == "__main__":
    unittest.main()
