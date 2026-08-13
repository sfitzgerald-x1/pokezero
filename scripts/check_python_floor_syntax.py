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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default=None, help="override the pyproject floor, e.g. py311")
    args = ap.parse_args()
    floor = args.floor or declared_floor()
    # N6: validated, not a free-text escape hatch. `--floor py99` made ruff exit 2, which (before
    # B1 above) printed the success line and returned 0 -- a one-flag silent-green switch.
    if not re.fullmatch(r"py3\d{1,2}", floor):
        print(
            f"floor {floor!r} is not a ruff target-version of the form py3NN. A target ruff does "
            "not recognise makes it exit 2, which is 'could not run', not 'clean'.",
            file=sys.stderr,
        )
        raise SystemExit(2)

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

    proc = subprocess.run(
        # `--isolated` -- the fifth silent-green path, and the only one reachable by a CORRECT
        # config rather than a typo. `force-exclude = true` alongside any `exclude` makes ruff skip
        # even explicitly-passed paths and exit 0, so:
        #   - the returncode check below never fires (ruff exited 0, not 2);
        #   - `tracked_files` is unchanged at 522, because it counts `git ls-files` output, not what
        #     ruff examined -- so a denominator pin would not catch it either;
        #   - the success line prints, so the workflow's shape grep is green.
        # Demonstrated on the real tree with the real historical defect restored: 41 diagnostics
        # became `invalid-syntax=0 ... every tracked .py file parses on py311`, exit 0.
        # `force-exclude = true` is the RECOMMENDED setting under pre-commit, not a mistake, so this
        # is the competent config rather than the incompetent one. `--isolated` is strictly stronger
        # than `--no-force-exclude`: it ignores every `[tool.ruff]` section, present or future, so
        # the gate is hermetic. It costs nothing, because `invalid-syntax` is emitted regardless of
        # rule selection.
        [ruff(), "check", "--isolated", f"--target-version={floor}", "--no-cache",
         "--output-format=concise", "--", *files],
        capture_output=True, text=True, cwd=REPO,
    )
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
