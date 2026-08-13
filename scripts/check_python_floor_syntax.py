#!/usr/bin/env python3
"""Fail if any tracked .py uses syntax newer than `requires-python` declares.

WHY THIS EXISTS. `scripts/c153_wide_negative_census.py` used a backslash inside an f-string
expression -- legal on 3.12+, a SyntaxError on 3.11. `pyproject.toml` declares
`requires-python = ">=3.11"`, so the file was in breach of the project's own floor. The
consequences were invisible in both directions:

  - CI pins 3.12 in all five jobs, so CI could not see it;
  - locally it failed COLLECTION for three test modules plus one script, which presented as
    "four files you have to work around" -- and the suite then silently measured 116 tests and
    1398 subtests fewer than it appeared to cover. Two stale artifacts rotted unnoticed inside
    those modules.

It went unnoticed for as long as it did because nothing anywhere compared the code against the
declared floor.

WHY NOT A PURE-PYTHON GUARD. There isn't one. PEP 701 removed the restriction from CPython's
parser, so `ast.parse(src, feature_version=(3, 11))` on a 3.12+ interpreter ACCEPTS the very
syntax that 3.11 rejects -- verified. Only a tool with its own version-aware parser can answer
this, which is why this shells out to ruff.

SCOPE. Deliberately narrow: only ruff's `invalid-syntax` diagnostics, which are "this cannot
parse on the target version". Ordinary lint findings are ignored, so this cannot become a
general style gate by accident. For scale: this exact invocation reports 2948 ordinary findings
on the current tree (RUF100 464, I001 454, UP045 258, UP037 242, ...), every one of them ignored
here. An earlier revision of this docstring said "the three pre-existing findings", which was
wrong by three orders of magnitude; behaviour was unaffected but the claim was not checkable.

WHAT THIS DOES NOT CATCH. It is a SYNTAX gate, not a floor gate. `from itertools import batched`
is valid syntax at `--target-version=py311` and raises ImportError on 3.11 -- verified. Nothing
here sees 3.12+ stdlib or API usage, only syntax the older parser rejects. The original defect
was syntactic, and so is the whole PEP 695/PEP 701 surface, but the gap is real and named.

Usage:  python scripts/check_python_floor_syntax.py [--floor py311]
Exit 0 = every tracked file parses on the declared floor. Exit 1 = at least one does not.
Exit 2 = the check could not run (no ruff, or no floor declared) -- never silently green.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def declared_floor() -> str:
    """Derive the target from pyproject, so the gate cannot drift from the declaration."""
    try:
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    except OSError as exc:
        # Exit 2, not a traceback. An unreadable pyproject raised FileNotFoundError out of here,
        # and an uncaught exception exits 1 -- which in this script's contract means "found a
        # violation". "Could not run" and "found a violation" must not be the same code.
        print(f"cannot read pyproject.toml ({exc}); the floor is undeclared", file=sys.stderr)
        raise SystemExit(2) from exc
    m = re.search(r'^requires-python\s*=\s*"[><=~^]*\s*(\d+)\.(\d+)', text, re.M)
    if not m:
        print("no requires-python in pyproject.toml -- nothing to check against", file=sys.stderr)
        raise SystemExit(2)
    return f"py{m.group(1)}{m.group(2)}"


def ruff() -> str:
    # The repo venv wins over PATH, so a LOCAL run uses whatever ruff the developer has installed
    # while CI uses the pinned one from the workflow. That asymmetry is deliberate -- a local run is
    # a convenience, and CI is the gate -- but it means a local green is weaker than a CI green.
    for cand in (REPO / ".venv/bin/ruff", Path("ruff")):
        path = shutil.which(str(cand)) or (str(cand) if Path(cand).exists() else None)
        if path:
            return path
    print(
        "ruff not found. This check cannot be done in pure Python -- PEP 701 removed the "
        "restriction from CPython's parser, so ast.parse(feature_version=...) accepts syntax the "
        "older interpreter rejects. Install ruff rather than skipping the check.",
        file=sys.stderr,
    )
    # SystemExit with a STRING exits 1 and prints it; only an int sets the code. The first cut
    # wrote SystemExit("2: ...") and therefore exited 1 -- so "could not run" was
    # indistinguishable from "found a violation", and the documented contract was simply false.
    raise SystemExit(2)


# Syntax that is INVALID at the given floor and valid at the next version, keyed by floor. Used by
# --self-test to prove ruff still detects a floor breach, at the floor the project actually declares.
#
# A hand-written table is unavoidable -- "a construct newer than version X" is not derivable -- but
# the FLOOR is not hardcoded, and an unknown floor exits 2 rather than silently attesting nothing.
# That distinction is the point: the previous kill-confirm hardcoded `py311` in the workflow, so
# raising `requires-python` to >=3.12 would have left it validating a capability the gate no longer
# exercises. Verified: at floor py312 the py311 fixture returns "All checks passed!".
def run_ruff_show_files(target: str, paths: list[str]) -> subprocess.CompletedProcess:
    """What ruff WILL check, honouring exclusions. Same flags as the real run, plus --show-files."""
    return _ruff(target, paths, extra=["--show-files"])


def run_ruff(target: str, paths: list[str]) -> subprocess.CompletedProcess:
    """The ONE ruff invocation. Every caller goes through here.

    B2: the argv used to be written out three times -- once in `main`, once in `self_test`, once in
    `check_fixture_table` -- so the kill-confirm validated a PARALLEL COPY of the flags rather than
    the gate's. Demonstrated: drop `--isolated` from `main` alone, leaving both self-test copies
    intact, add `[tool.ruff] force-exclude = true`, restore the historical defect -> the gate prints
    "every tracked .py file parses on py311" and exits 0 AND the self-test prints "self-test OK" and
    exits 0. Both CI steps green, 41 diagnostics suppressed. A guard that validates a copy of the
    thing under test is not a guard, which is this PR's own thesis applied to its own kill-confirm.

    `--color=never`, and the environment scrubbed of RUFF_*: `--isolated` makes the gate hermetic in
    CONFIG but not in ENVIRONMENT. Three variables survived it and produced a full silent green with
    the historical defect present -- ruff still exits 1, so the returncode branch does not fire, but:
      FORCE_COLOR=1 / CLICOLOR_FORCE=1  ANSI bytes land between the ': ' and 'invalid-syntax', so
                                        the diagnostic regex stops matching. NO_COLOR does NOT win.
      RUFF_OUTPUT_FILE=<path>           diagnostics go to that file; stdout is empty.
    In each case `tracked_files` stayed at 522, so a denominator pin could not see it either, and
    the workflow's shape grep was green. `--color=never` fixes the first two and the env scrub fixes
    the third; both are needed -- verified independently, 41 diagnostics recovered only with both.
    """
    return _ruff(target, paths)


def _ruff(target: str, paths: list[str], *, extra: list[str] | None = None):
    """The single argv. Both runners and the self-test go through here, so none can drift."""
    scrubbed = {k: v for k, v in os.environ.items() if not k.startswith("RUFF_")}
    return subprocess.run(
        [ruff(), "check", "--isolated", "--color=never", f"--target-version={target}",
         "--no-cache", *(extra or []), "--output-format=concise", "--", *paths],
        capture_output=True, text=True, env=scrubbed,
    )


KNOWN_BAD_BY_FLOOR = {
    # Each value must be a SyntaxError at its key and valid at the next version. Enforced below by
    # `check_fixture_table`, not by these comments -- the first version of this table had py313
    # mapped to `class C[T = int]` and PEP 696 landed in 3.13, so the fixture was VALID at py313.
    # I wrote it from memory ("PEP 696 -> 3.13") instead of deriving it, and duplicated the py312
    # feature under a second key. Exactly the defect this whole gate exists to close, in the gate's
    # own self-test data.
    "py311": "type Alias = int\n",                              # PEP 695 type alias: 3.12
    "py312": "def f[T = int](x: T) -> T:\n    return x\n",       # PEP 696 defaults: 3.13
    "py313": "try:\n    pass\nexcept ValueError, TypeError:\n    pass\n",  # PEP 758: 3.14
}


def _next_version(floor: str) -> str:
    """py312 -> py313. Used to prove a fixture is a FLOOR breach, not just broken syntax."""
    return f"py3{int(floor[3:]) + 1}"


def check_fixture_table() -> list[str]:
    """Every fixture must redden at its own floor AND be accepted at the next version.

    The second half is what makes an entry meaningful. Without it, a fixture that is simply invalid
    Python everywhere would satisfy the self-test while proving nothing about version detection --
    and a fixture valid at its own floor would go unnoticed until someone raised the floor and got
    "the gate cannot be trusted", blaming the gate rather than this table.

    Returns a list of problems; empty means the table is sound.
    """
    import tempfile

    problems: list[str] = []
    for floor, source in sorted(KNOWN_BAD_BY_FLOOR.items()):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "fixture.py"
            probe.write_text(source, encoding="utf-8")
            at_floor = run_ruff(floor, [str(probe)])
            at_next = run_ruff(_next_version(floor), [str(probe)])
        if at_floor.returncode != 1 or not re.search(
            r":\d+:\d+: invalid-syntax:", at_floor.stdout
        ):
            problems.append(
                f"{floor}: the fixture is ACCEPTED at {floor} (ruff exit {at_floor.returncode}), so "
                f"it cannot detect a {floor} breach. It must be a SyntaxError at {floor}."
            )
        if at_next.returncode != 0:
            problems.append(
                f"{floor}: the fixture is ALSO rejected at {_next_version(floor)}, so it is not a "
                f"version boundary -- it is just invalid Python, and would pass the self-test while "
                f"proving nothing about version detection."
            )
    return problems


def self_test(floor: str) -> int:
    """Prove ruff still reddens on syntax invalid at THIS floor. Exit 2 if it cannot be proven.

    Without this, a ruff rename or output-format change returns the gate to permanent green with no
    signal: the script would exit 0 and still print its success line, so the workflow's shape grep
    cannot see it. A pinned version stops accidental drift; this stops silent drift.
    """
    import tempfile

    # Validate the WHOLE table, not just this floor's entry: a wrong entry for a floor nobody has
    # raised to yet is still wrong, and CI is where it should surface rather than at the moment
    # somebody edits pyproject.
    problems = check_fixture_table()
    if problems:
        print("the self-test fixture table is unsound:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    source = KNOWN_BAD_BY_FLOOR.get(floor)
    if source is None:
        print(
            f"no known-bad fixture for floor {floor}; the gate cannot be self-tested and is "
            f"therefore unverified. Add an entry to KNOWN_BAD_BY_FLOOR for {floor} -- syntax that "
            "is a SyntaxError at that version and valid at the next one.",
            file=sys.stderr,
        )
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        # Outside the repo, so it can never be linted, imported or collected. `--isolated` means it
        # shares the gate's (empty) config scope regardless of where it lives -- which is what makes
        # a temp location correct here rather than accidentally wrong.
        probe = Path(tmp) / "known_bad.py"
        probe.write_text(source, encoding="utf-8")
        proc = run_ruff(floor, [str(probe)])
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 1:
        print(
            f"ruff exited {proc.returncode} on input that cannot parse at {floor}; expected 1. The "
            "gate cannot be trusted -- it would report green without checking anything.",
            file=sys.stderr,
        )
        return 2
    if not re.search(r":\d+:\d+: invalid-syntax:", proc.stdout):
        print(
            "ruff no longer emits an 'invalid-syntax' diagnostic in the expected format. The gate "
            "greps for that token, so it would be permanently green.",
            file=sys.stderr,
        )
        return 2
    print(f"self-test OK: ruff rejects {floor}-invalid syntax at --target-version={floor}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default=None, help="override the pyproject floor, e.g. py311")
    ap.add_argument(
        "--print-floor", action="store_true",
        help="print the derived floor and exit, so callers need not hardcode it",
    )
    ap.add_argument(
        "--self-test", action="store_true",
        help="prove ruff still reddens on syntax invalid AT THE DERIVED FLOOR, then exit",
    )
    args = ap.parse_args()
    floor = args.floor or declared_floor()
    # N6: validated, not a free-text escape hatch. `--floor py99` made ruff exit 2, which (before
    # B1 above) printed the success line and returned 0 -- a one-flag silent-green switch.
    if args.print_floor:
        # So nothing downstream hardcodes what this script derives.
        print(floor)
        return 0
    if not re.fullmatch(r"py3\d{1,2}", floor):
        print(
            f"floor {floor!r} is not a ruff target-version of the form py3NN. A target ruff does "
            "not recognise makes it exit 2, which is 'could not run', not 'clean'.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.self_test:
        return self_test(floor)

    # `-z` and split on NUL, NOT `.split()`. Whitespace splitting shreds any path containing a
    # space into two nonexistent paths: ruff reports nothing for either, the count is INFLATED by
    # one, and the real file is never checked. Fail-open, and invisible to a denominator pin
    # because the number went up. Non-ASCII paths fail the same way -- `ls-files` C-quotes them
    # unless `-z` is used. No such path exists in the tree today, so this is latent rather than
    # live, but it is the exact fail-open class this gate exists to close, and the correct idiom
    # is already used elsewhere in the repo (tests/test_roll_enumeration_scope.py).
    try:
        listing = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "*.py"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        # Same reason as declared_floor: `check=True` raised CalledProcessError, which exits 1 and
        # therefore claimed a violation when the truth was that nothing was enumerated.
        print(f"cannot list tracked files ({exc}); nothing was measured", file=sys.stderr)
        raise SystemExit(2) from exc
    files = [f for f in listing.split("\0") if f]
    if not files:
        print("no tracked .py files found -- the check measured nothing", file=sys.stderr)
        raise SystemExit(2)

    # RECONCILE THE DENOMINATOR. `tracked_files` counts what `git ls-files` returned; it has never
    # said anything about what ruff actually opened, so the gate could not distinguish "522 files
    # parse" from "N parse and 522-N were silently skipped". `--show-files` lists exactly the paths
    # ruff will check, honouring every exclusion, so comparing its count against the paths passed in
    # closes the class rather than one instance of it.
    #
    # This is what finally kills the config-silencing family. Measured on 60 paths with
    # `[tool.ruff] force-exclude = true, exclude = ["scripts"]`: without `--isolated` ruff lists
    # 2 of 60; with it, 60 of 60. An earlier attempt at this check placed a probe file inside the
    # repo and required it to be flagged -- insufficient, because a hostile config excludes some
    # directory and the probe sits in another, so it survived the very mutant it was written for.
    # Counting what ruff will examine does not depend on guessing where the exclusion points.
    shown = run_ruff_show_files(floor, files)
    if shown.returncode != 0:
        print(shown.stdout, end="")
        print(shown.stderr, end="", file=sys.stderr)
        print(
            f"could not enumerate what ruff would check (exit {shown.returncode}); the denominator "
            "is unverifiable, so this is not a pass.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    examined = len([line for line in shown.stdout.splitlines() if line.strip()])
    if examined != len(files):
        print(
            f"ruff would check {examined} of the {len(files)} tracked .py files passed to it, so "
            f"{len(files) - examined} would be silently SKIPPED and reported as clean. A config "
            "section is excluding them, or a flag that prevents that is missing.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    proc = run_ruff(floor, files)
    # B1: ruff's OWN failure must not read as clean. ruff exits 0 = no findings, 1 = findings,
    # >=2 = it could not run (bad rule selector, unreadable config, unknown target). On >=2 it
    # writes to stderr and produces ZERO bytes of stdout -- so without this check `bad` is empty,
    # the success line prints, and the gate returns 0. Demonstrated three ways: `--floor py99`, a
    # `requires-python = ">=3.16"`, and a typo'd rule in a `[tool.ruff.lint] select`. The third is
    # the realistic one: anyone adding a ruff config section with a typo disarms this gate
    # permanently, and the workflow's output-shape grep cannot tell, because the shape is right.
    if proc.returncode >= 2:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        print(
            f"ruff could not run (exit {proc.returncode}), so nothing was checked. This is not a "
            "pass: a gate that cannot run has not measured anything.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # E902 is "ruff could not READ this file", and an unread file is an UNCHECKED file. Counting
    # only `invalid-syntax` made those indistinguishable from clean ones: a scratch repo with a
    # tracked dangling symlink and a chmod-000 file containing `type B = int` -- a real 3.11
    # SyntaxError -- reported `invalid-syntax=0`, printed the success line, and exited 0 while ruff
    # was saying `E902 No such file or directory` and `E902 Permission denied`.
    #
    # This is the CLASS the whitespace-split bug was one instance of. That fix removed the cause
    # (paths shredded into nonexistent ones, which ruff then reported as E902) and left the class
    # standing. The structural gap it exposes: `tracked_files` is asserted from `git ls-files` and
    # never reconciled against what ruff actually opened, so the gate could not distinguish
    # "522 files parse" from "N parse and 522-N were never read". E902 is ruff telling us exactly
    # that, per file, and exit 2 -- "could not run" -- is this contract's own answer to it.
    unread = [l for l in proc.stdout.splitlines() if re.search(r":\s*E902\b", l)]
    if unread:
        print(
            f"ruff could not READ {len(unread)} file(s), so they were NOT checked. This is not a "
            "pass:",
            file=sys.stderr,
        )
        for line in unread[:20]:
            print(f"  {line}", file=sys.stderr)
        if len(unread) > 20:
            print(f"  ... and {len(unread) - 20} more", file=sys.stderr)
        raise SystemExit(2)

    # N5: match the RULE FIELD, not the line. `"invalid-syntax" in l` also matches a diagnostic
    # about a FILE whose name contains the token -- verified: `x_invalid-syntax_y.py` with an
    # unused import counted as a violation.
    bad = [l for l in proc.stdout.splitlines() if re.search(r":\s*invalid-syntax:", l)]
    print(f"floor={floor} tracked_files={len(files)} invalid-syntax={len(bad)}")
    if bad:
        print(f"\n{len(bad)} diagnostic(s) -- these files cannot be parsed on {floor}:")
        for line in bad[:40]:
            print(f"  {line}")
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more")
        print(
            f"\npyproject declares requires-python >= {floor[2]}.{floor[3:]}, so this is a breach "
            "of the project's own floor. CI pins a newer interpreter and cannot see it; the cost "
            "lands on anyone running the declared minimum, as silent collection failures."
        )
        return 1
    print(f"every tracked .py file parses on {floor}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
