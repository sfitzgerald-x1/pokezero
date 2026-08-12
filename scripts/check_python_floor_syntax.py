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
general style gate by accident, and it stays green on the three pre-existing findings in the
tree today.

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
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^requires-python\s*=\s*"[><=~^]*\s*(\d+)\.(\d+)', text, re.M)
    if not m:
        print("no requires-python in pyproject.toml -- nothing to check against", file=sys.stderr)
        raise SystemExit(2)
    return f"py{m.group(1)}{m.group(2)}"


def ruff() -> str:
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

    files = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    if not files:
        print("no tracked .py files found -- the check measured nothing", file=sys.stderr)
        raise SystemExit(2)

    proc = subprocess.run(
        [ruff(), "check", f"--target-version={floor}", "--no-cache",
         "--output-format=concise", "--", *files],
        capture_output=True, text=True, cwd=REPO,
    )
    bad = [l for l in proc.stdout.splitlines() if "invalid-syntax" in l]
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
