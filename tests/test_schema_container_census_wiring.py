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

  4. EXECUTION -- the `The census wiring pins` step of that same job runs THIS module, with exact
     `Ran N tests` and `^OK$` guards, and `test_the_workflow_test_count_guard_matches_this_module`
     derives N from the loader so the guard cannot go stale without a LOCAL failure. That
     self-check is what makes the omission impossible for this module rather than merely fixed
     once; tests/test_harness_digest_provenance.py, the precedent this file is modelled on, has
     carried it since #1163 and copying the pins without it is how the property went missing.

The behavioural half runs against MINIMAL synthetic trees rather than the repo, so each assertion
has a known answer that does not move when a container is added to src/. The class pin is
substituted per fixture for the same reason -- these tests are about the mechanism, not about the
current contents of the spec file.

Assertions about a STEP read `_step_block`, not the whole job. Two of them were satisfiable by a
sibling step before that helper existed -- `assertIn('rc=$?', block)` was satisfied by the
kill-confirm's own `rc=$?`, so deleting the verdict step's went uncaught (measured green at
8d32dcb2) -- which is the `fetch-depth`-matched-its-own-comment defect at a different granularity.
"""

from __future__ import annotations

import ast
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "engine-fidelity-gates.yml"
CENSUS = REPO / "scripts" / "schema_container_census.py"
SPEC = REPO / "tests" / "data" / "schema_drill_schema_containers.txt"
JOB = "schema-container-census"
VERDICT_STEP = "Schema-container census"
KILL_CONFIRM_STEP = "The census reddens on known-bad input"
WIRING_STEP = "The census wiring pins"

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


def _step_block(job: str, step_name: str) -> str:
    """The YAML text of ONE named step of a job, comments removed.

    Assertions about a step's body must not be satisfiable by a sibling step's body. Two of this
    file's assertions were, before this helper existed: see `_has_line`.
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
    return "\n".join(line for line in body if not line.strip().startswith("#"))


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

        keys: list[str] = []
        for line in _job_block(JOB).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:]
            keys.append(stripped.split(":")[0].strip().strip("\"'"))

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

    def test_the_job_actually_runs_the_census(self) -> None:
        step = _step_block(JOB, VERDICT_STEP)
        self.assertTrue(
            _has_line(step, "python scripts/schema_container_census.py"),
            "the verdict step no longer invokes the census on the checked-out tree; everything "
            "else here asserts nothing. Note this is a WHOLE-LINE match: the census must be the "
            "command, not a substring of a longer pipeline whose exit code belongs to something "
            "else, and the flag list is pinned too -- `--check` was removed for being a no-op "
            "whose --help claimed a behaviour, so passing it here would be an argparse exit 2",
        )

    def test_the_verdict_is_the_exit_code_not_a_grep_of_its_own_output(self) -> None:
        """34 of `mass-gate`'s 44 steps assert on a log their own step wrote via `tee`, a shape
        held together only by `set -o pipefail` under `bash -e`. The VERDICT step must not join
        them: a success-line grep can be satisfied by an `echo`, whereas an exit code cannot.

        Scoped to that one step, not to the job. The job now also runs this module and greps its
        `Ran N tests` / `^OK$` log, which is a different thing -- that log is written by a separate
        process whose invocation is pinned below -- and a job-wide `assertNotIn("tee ")` would have
        forbidden it. Scoping is also what makes the `rc=$?` assertion mean anything: over the whole
        job it was satisfied by the KILL-CONFIRM step's own `rc=$?`, so deleting the one here was
        not caught. Found by mutation, same family as the `fetch-depth` comment match.
        """

        step = _step_block(JOB, VERDICT_STEP)
        self.assertNotIn(
            "tee ", step, "the verdict step now tees its output, inviting a self-grep verdict"
        )
        self.assertTrue(
            _has_line(step, "rc=$?"),
            "the census's exit code is no longer captured in the verdict step, so it cannot be "
            "the verdict",
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
                self.assertIn(
                    "exit 1",
                    _guard_body(step, needle),
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
        self.assertIn(
            "exit 1",
            _guard_body(step, '[ "${fail}" -ne 0 ]'),
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
        self.assertIn(
            "exit 1",
            _guard_body(block, '[ "${head_commit}" = "${HEAD_SHA}" ]'),
            "censusing the branch tip no longer fails",
        )
        self.assertIn(
            "exit 1",
            _guard_body(block, "if ! second_parent=$(git rev-parse --verify --quiet 'HEAD^2')"),
            "a single-parent checkout no longer fails, so `ref: main` would go green",
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
        self.assertIn(
            "exit 1",
            _guard_body(block, '[ "${CENSUS}" != "success" ]'),
            "the census branch of gate-status no longer exits nonzero",
        )

    def test_a_job_actually_runs_this_module(self) -> None:
        """THE PIN THAT MAKES EVERY OTHER PIN IN THIS FILE MEAN ANYTHING.

        At 8d32dcb2 this module executed NOWHERE. `grep -rn test_schema_container_census_wiring
        .github/ scripts/` returned zero hits, and this workflow names test modules one by one --
        35 explicit `python -m unittest tests.<module>` invocations, no `unittest discover`, no
        pytest anywhere in .github/ -- so a new module is invisible to CI by default. Every
        assertion here was therefore local-only: a job-level `if: false`, a `continue-on-error`,
        removal from `gate-status`'s `needs:`, an `exit 1` flipped to `exit 0` or an emptied pin
        table would each have landed on main green, which is the exact defect this PR exists to
        close, reproduced one level up.

        The `Ran N tests` and `^OK$` guards read a LOG, and a log can be forged: replacing the
        `run:` body with `echo 'Ran N tests'; echo OK` satisfies both greps. Pinning the invocation
        is what makes them mean anything.
        """

        step = _step_block(JOB, WIRING_STEP)
        self.assertIn(
            "python -m unittest tests.test_schema_container_census_wiring",
            step,
            "the job no longer runs this module; every pin in this file is inert again",
        )
        # Both log guards must exit nonzero, for the same reason gate-status's branch must: `exit 1`
        # -> `exit 0` leaves the grep, the message and the whole shape of the guard intact while the
        # step passes over a shrunk, failing or forged suite.
        for needle in ("Ran ", "'^OK$'"):
            with self.subTest(guard=needle):
                self.assertIn(
                    "exit 1",
                    _guard_body(step, needle),
                    f"the {needle} guard no longer exits nonzero",
                )

    def test_the_workflow_test_count_guard_matches_this_module(self) -> None:
        """THE SELF-CHECK, and the property this module was missing while its precedent had it.

        `tests/test_harness_digest_provenance.py` asserts that the workflow's exact `Ran 28 tests`
        guard matches its own loader count, which is what makes forgetting that guard a LOCAL
        failure rather than a red CI run after the push. The guard lives in YAML, so no local
        `unittest` or `pytest` run can otherwise see it, and a module that grows without its count
        following is how an approved PR has reddened CI in this repo before.

        Read from `_job_code`, not `_job_block`, which is stricter than the precedent: a comment in
        the job that happened to quote the right number must not satisfy this. That is the same
        comment-matches-its-own-prose defect `_job_code` exists for.
        """

        count = unittest.defaultTestLoader.loadTestsFromModule(
            sys.modules[__name__]
        ).countTestCases()
        self.assertIn(
            f"Ran {count} tests",
            _job_code(JOB),
            f"this module now has {count} tests; the workflow's exact grep guard for {JOB} still "
            "names a different number, so CI would go red on a green suite",
        )
        # Anti-vacuity: a loader that found nothing would make the assertion above pass against a
        # `Ran 0 tests` guard nobody would notice was wrong.
        self.assertGreater(count, 1, "anti-vacuity: the loader found no tests in this module")


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
