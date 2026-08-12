"""The schema-default conflation is fail-closed at AUTHORSHIP, not at the next rotation.

Reading the process-wide observation-schema default where a specific version or property is
meant is the defect class that made a one-line default rotation break 94 tests, and that hid
two real production bugs (#1227 `token_count`, #1228 the feature widths) for two schema
generations. Python has no way to make a module constant unreadable, so the enforcement is
here: the set of sites reaching the default is derived from the tree and compared against a
committed allowlist.

The allowlist only ever shrinks. A NEW site fails this test in the PR that introduces it,
which is the whole point -- the alternative is discovering it during the v5 rotation, which is
how the v4 rotation went.
"""
from __future__ import annotations

import json
from collections import Counter
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# tests/data, not corpus/: corpus/ is gitignored, so an allowlist there is invisible to
# everyone else and the gate silently cannot run. Caught by `git add` refusing the path.
ALLOWLIST = REPO / "tests" / "data" / "schema_default_allowlist.json"
LEDGER = REPO / "scripts" / "schema_default_ledger.py"


def _derive() -> list[dict]:
    """Re-derive from the tree. Never read a cached count -- that is the error being retired."""
    # 3.12+: the script only parses, and one tracked file uses a backslash inside an f-string
    # expression, which is a SyntaxError on 3.11. It exits 2 rather than under-reporting.
    proc = subprocess.run(
        [sys.executable, str(LEDGER), "--json"], capture_output=True, text=True
    )
    if proc.returncode == 2:
        raise unittest.SkipTest(
            "ledger reported UNPARSED files under this interpreter, so the denominator is "
            "incomplete; run the suite on 3.12+. Skipping is correct here ONLY because the "
            "script fails loudly -- a silently short count would not be caught."
        )
    if proc.returncode != 0:
        raise AssertionError(f"ledger derivation failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def _key(row: dict) -> str:
    # File+kind+owner, NOT line: a row must survive unrelated line drift, or the gate becomes
    # noise and gets muted. Owner is the enclosing def, which is stable under edits above it.
    # NO line in the key. Keying on the line made the gate interpreter-dependent: `ast` reports
    # a different `lineno` for the same multi-line call across Python versions, so an allowlist
    # generated on 3.14 failed on CI's 3.12 with an off-by-one (`fallback_replay.py::...::112`
    # vs `::111`). That was my own fix for bypassability, and it traded a real hole for a
    # portability bug.
    #
    # Bypassability is instead closed by COUNTING per key (see below): file+owner+kind collapsed
    # 97 rows to 74 distinct keys, so a set comparison let a diff migrate one site and add
    # another at the same key. A multiset does not.
    return f"{row['file']}::{row['owner']}::{row['kind']}"


class SchemaDefaultLedgerTest(unittest.TestCase):
    def test_no_new_site_reaches_the_global_default(self) -> None:
        derived = Counter(_key(r) for r in _derive())
        allowed = Counter(_key(r) for r in json.loads(ALLOWLIST.read_text()))
        # A multiset difference: a key whose COUNT grew is a new site even if the key existed.
        new = sorted((derived - allowed).elements())
        self.assertEqual(
            new,
            [],
            "new site(s) reaching the global observation-schema default:\n  "
            + "\n  ".join(new)
            + "\n\nSay what you need instead of taking the default:\n"
            "  - a specific version  -> OBSERVATION_SCHEMA_VERSION_V2_2 (etc.)\n"
            "  - a schema property   -> schema_with(transition_region=True)\n"
            "If this site is a genuine default reader, add it to "
            "tests/data/schema_default_allowlist.json WITH a justification in the PR body; there "
            "are only five, and each is load-bearing.",
        )

    def test_the_allowlist_only_shrinks(self) -> None:
        """Retired rows must be REMOVED from the allowlist, or the burndown cannot converge."""
        derived = Counter(_key(r) for r in _derive())
        allowed = Counter(_key(r) for r in json.loads(ALLOWLIST.read_text()))
        stale = sorted((allowed - derived).elements())
        self.assertEqual(
            stale,
            [],
            f"{len(stale)} allowlist row(s) no longer exist in the tree. Migrating a site is "
            "only half the change -- drop its row so the remaining count is the real one:\n  "
            + "\n  ".join(stale[:20]),
        )

    def test_the_burndown_count_is_pinned(self) -> None:
        """A visible number, so progress and regression are both legible in the diff.

        This is the figure the whole effort is measured by, and it is DERIVED here rather than
        quoted: four different values were reported for it before the ledger existed (6, ~4,
        97, 94), every one of them without an established denominator.
        """
        self.assertEqual(
            len(_derive()),
            len(json.loads(ALLOWLIST.read_text())),
            "the derived site count and the allowlist have diverged; regenerate with "
            "`python3.12 scripts/schema_default_ledger.py --json > "
            "tests/data/schema_default_allowlist.json` and justify every added row.",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
