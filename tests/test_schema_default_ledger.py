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
import re
from collections import Counter
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# tests/data, not corpus/: corpus/ is gitignored, so an allowlist there is invisible to
# everyone else and the gate silently cannot run. Caught by `git add` refusing the path.
ALLOWLIST = REPO / "tests" / "data" / "schema_default_allowlist.json"
# The most rows the allowlist has ever legitimately held. Only ever lowered.
HIGH_WATER_MARK = 391
LEDGER = REPO / "scripts" / "schema_default_ledger.py"


def _derive() -> list[dict]:
    """Re-derive from the tree. Never read a cached count -- that is the error being retired."""
    proc = subprocess.run(
        [sys.executable, str(LEDGER), "--json"], capture_output=True, text=True
    )
    if proc.returncode == 2:
        raise unittest.SkipTest(
            "the ledger reported UNPARSED file(s), so the denominator is incomplete and there "
            "is nothing valid to compare. Fix the unparsable file. (Until #1239 this fired "
            "routinely because one tracked file could not be parsed on 3.11 and the advice was "
            "'use 3.12+'; that is no longer true, and on any interpreter a SKIP here now means a "
            "genuinely broken file rather than a known-tolerated one.)"
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
    # `unclosed` IS part of the key. Without it, a row could silently gain routes: deleting both
    # width kwargs from an existing call left N, the key count and every gate test unchanged, while
    # that call went from defaulting one route to defaulting three. That is a regression of exactly
    # the shape this ledger cites as its reason to exist (41 of the rows the any-of bug hid pin a
    # width and default the schema), and it was invisible because `unclosed` was recorded in the
    # allowlist and then excluded from every comparison.
    unclosed = ",".join(row.get("unclosed", ()))
    return f"{row['file']}::{row['owner']}::{row['kind']}::{unclosed}"


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
            "If this site is a genuine default reader -- it ANSWERS 'nobody said' rather than "
            "consuming the answer -- add it to tests/data/schema_default_allowlist.json with a "
            "justification in the PR body. Most sites are NOT that: the allowlist is a "
            "grandfathered snapshot of existing exposure, not a list of blessed readers, and it "
            "is expected to shrink. (An earlier version of this message claimed 'there are only "
            "five, and each is load-bearing' while the allowlist held 202 rows -- actively "
            "misleading to whoever read it next.)",
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

    def test_the_allowlist_can_only_shrink_from_its_high_water_mark(self) -> None:
        """A CEILING, not just an equality. The other tests compare the tree to the allowlist, so
        adding a site AND regenerating the allowlist keeps them agreeing and passes -- which is
        exactly the move that would quietly grow the exposure this file exists to bound. The
        high-water mark makes growth an explicit, reviewable edit to this number.
        """
        rows = len(json.loads(ALLOWLIST.read_text()))
        self.assertLessEqual(
            rows,
            HIGH_WATER_MARK,
            f"the allowlist grew to {rows} rows, above the {HIGH_WATER_MARK} recorded here. "
            "Migrating sites is the intended direction. If a NEW default reader is genuinely "
            "legitimate, raise this number in the same commit and justify it in the PR body -- "
            "regenerating the allowlist alone keeps every other test in this file green.",
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
            "`.venv/bin/python scripts/schema_default_ledger.py --json > "
            "tests/data/schema_default_allowlist.json` and justify every added row.",
        )


class LedgerDocstringIsHeldToTheToolTest(unittest.TestCase):
    """The per-kind counts in the ledger's own docstring must equal what the ledger derives.

    Not a style check. That docstring listed two kinds (`implicit-spec`, `implicit-cfg`) that the
    code had stopped emitting, for long enough that a reviewer used them to reason about output the
    tool could not produce; the fix then replaced them with nine fresh numbers pinned by nothing,
    which is the same arrangement that staled the first time. Prose that states a derived figure is
    either checked against the derivation or it is a comment about the past.
    """

    def _docstring_counts(self) -> dict[str, int]:
        """Parse `Name  <count>` pairs out of the module docstring, without importing it."""
        import ast as _ast

        doc = _ast.get_docstring(_ast.parse(LEDGER.read_text(encoding="utf-8"))) or ""
        found: dict[str, int] = {}
        for name, count in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*) \*?\s+(\d+)\b", doc):
            if name in ("bare", "default", "implicit"):
                continue
            found[name] = int(count)
        return found

    def _derived_counts(self) -> dict[str, int]:
        counts = Counter(
            row["kind"].split(":", 1)[1] for row in _derive() if row["kind"].startswith("implicit:")
        )
        # A modelled surface with no open sites must still be checkable as 0.
        for surface in self._surfaces():
            counts.setdefault(surface, 0)
        return dict(counts)

    def _surfaces(self) -> list[str]:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_ledger_surfaces", LEDGER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return sorted(module.SURFACES)

    def test_the_docstring_table_lists_every_surface_with_the_derived_count(self) -> None:
        derived = self._derived_counts()
        documented = self._docstring_counts()
        self.assertEqual(
            sorted(documented), sorted(derived),
            "the docstring's surface table and the derived surfaces disagree.\n"
            f"  documented: {sorted(documented)}\n  derived:    {sorted(derived)}\n"
            "Every surface must appear -- including one with 0 open sites, which is the goal state "
            "and must stay named so a re-opened call site lands in its bucket.",
        )
        for surface in sorted(derived):
            with self.subTest(surface=surface):
                self.assertEqual(
                    documented[surface], derived[surface],
                    f"the docstring says {surface} has {documented[surface]} sites; the tool "
                    f"derives {derived[surface]}.",
                )

    def test_the_docstring_states_the_surface_count_it_actually_has(self) -> None:
        doc = LEDGER.read_text(encoding="utf-8")
        n = len(self._surfaces())
        words = {7: "SEVEN", 8: "EIGHT", 9: "NINE"}
        self.assertIn(
            words.get(n, str(n)), doc.split('"""')[1],
            f"there are {n} surfaces; the docstring does not say so. The previous text said "
            "'seven derived surfaces' while listing an alternate constructor as derived and "
            "omitting a derived surface with no open sites.",
        )

    RETIRED_KINDS = ("implicit-spec", "implicit-cfg")

    def test_no_retired_kind_is_emitted_or_recorded(self) -> None:
        """The precise half: a retired name must never be a live `kind` VALUE.

        This is the check that actually binds. Prose about a retired name is harmless; a retired
        name in the emitted output or in the committed allowlist means the vocabulary the gate
        compares on has forked from the vocabulary the tool produces.
        """
        emitted = {row["kind"] for row in _derive()}
        recorded = {row["kind"] for row in json.loads(ALLOWLIST.read_text())}
        for retired in self.RETIRED_KINDS:
            with self.subTest(kind=retired):
                self.assertNotIn(retired, emitted, f"the ledger emits the retired kind {retired}")
                self.assertNotIn(
                    retired, recorded, f"the allowlist records the retired kind {retired}"
                )

    def test_a_retired_kind_is_only_ever_mentioned_as_retired(self) -> None:
        """The prose half: a mention must carry the fact that it is dead.

        Deliberately a vocabulary test rather than a file exclusion. Excluding the files that
        currently mention these names is how the stale table survived -- the mention was in the
        one file nobody re-derived. Requiring the retirement to be stated next to the name means a
        future reader cannot pick the name up and use it, which is what happened last time.
        """
        vocabulary = ("retired", "retires", "retirement", "stopped emitting", "no longer")
        # A WINDOW, not the matching line. Prose wraps: the first cut of this test checked only the
        # line carrying the name and failed on a sentence whose "stopped emitting" landed on the
        # next line. A guard that forces authors to fit a justification onto one line is a guard
        # that gets reworded around rather than satisfied.
        window = 2
        for retired in self.RETIRED_KINDS:
            with self.subTest(kind=retired):
                paths = subprocess.run(
                    ["git", "-C", str(REPO), "grep", "-l", retired,
                     "--", "*.py", "*.md", "*.yml", "*.yaml", "*.json"],
                    capture_output=True, text=True,
                ).stdout.split()
                unmarked = []
                for rel in paths:
                    lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
                    for i, line in enumerate(lines):
                        if retired not in line:
                            continue
                        context = " ".join(
                            lines[max(0, i - window):i + window + 1]
                        ).lower()
                        if not any(v in context for v in vocabulary):
                            unmarked.append(f"{rel}:{i + 1}: {line.strip()}")
                self.assertEqual(
                    unmarked, [],
                    f"{retired} is named with no statement within {window} lines that it is "
                    f"retired, so it reads as a live kind:\n  " + "\n  ".join(unmarked),
                )


class LedgerSeesEverySpellingTest(unittest.TestCase):
    """Every way to spell a read of the global default must produce a row.

    This class exists because the same defect recurred three times: the ledger's denominator was
    blind to a SPELLING, so new default reads could be added with N unchanged and the authorship
    gate green. Round 2 was any-of surface matching; round 5 was `... as ALIAS`; round 6 was a
    dotted base (`pokezero.observation.GLOBAL`) and the public bare `import pokezero`.

    Rounds 2 and 5 were each fixed and then verified ONCE, by hand, and pinned by nothing -- which
    is the actual root cause of the recurrence, not any individual missed spelling. A hole that no
    test holds shut reopens the next time the matcher is touched. So the enumeration lives here.
    """

    # (label, source). Each probe must yield exactly `expected` rows.
    PROBES = [
        ("from-import, plain",
         "from pokezero.observation import OBSERVATION_SCHEMA_VERSION\n"
         "def f():\n    return OBSERVATION_SCHEMA_VERSION\n", 1),
        ("from-import, aliased  (round-5 escape)",
         "from pokezero.observation import OBSERVATION_SCHEMA_VERSION as SV\n"
         "def f():\n    return SV\n", 1),
        ("spec from-import, aliased",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC as SP\n"
         "def f():\n    return SP.numeric_feature_count\n", 1),
        ("module import, aliased",
         "import pokezero.observation as O\n"
         "def f():\n    return O.OBSERVATION_SCHEMA_VERSION\n", 1),
        ("module import, dotted base  (round-6 escape A)",
         "import pokezero.observation\n"
         "def f():\n    return pokezero.observation.OBSERVATION_SCHEMA_VERSION\n", 1),
        ("bare `import pokezero`, package attr  (round-6 escape B: the PUBLIC spelling)",
         "import pokezero\n"
         "def f():\n    return pokezero.OBSERVATION_SCHEMA_VERSION\n", 1),
        ("bare `import pokezero`, base never imported by name",
         "import pokezero\n"
         "def f():\n    return pokezero.observation.OBSERVATION_SCHEMA_VERSION\n", 1),
        ("arbitrarily deep base",
         "import pokezero\n"
         "def f():\n    return pokezero.a.b.c.DEFAULT_REPLAY_OBSERVATION_SPEC\n", 1),
        ("`from pokezero import <mod>` outside the old hand-listed pair",
         "from pokezero import replay\n"
         "def f():\n    return replay.OBSERVATION_SCHEMA_VERSION\n", 1),
        ("two reads on one line are two sites",
         "import pokezero\n"
         "def f():\n"
         "    return (pokezero.OBSERVATION_SCHEMA_VERSION, pokezero.OBSERVATION_SCHEMA_VERSION)\n",
         2),
    ]

    # Spellings that must NOT produce a row. Without these the test is satisfied by a matcher that
    # flags every attribute access, which would inflate the denominator instead of measuring it.
    NEGATIVES = [
        ("a same-named attribute on an unrelated object",
         "import numpy\n"
         "def f():\n    return numpy.OBSERVATION_SCHEMA_VERSION\n"),
        ("the per-version names, deliberately not counted",
         "from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V4\n"
         "def f():\n    return OBSERVATION_SCHEMA_VERSION_V4\n"),
        ("a local variable that merely shares the name's shape",
         "def f():\n    OBSERVATION_SCHEMA_VERSION_LOCAL = 1\n"
         "    return OBSERVATION_SCHEMA_VERSION_LOCAL\n"),
        ("a string mentioning the constant",
         "def f():\n    return 'OBSERVATION_SCHEMA_VERSION'\n"),
    ]

    def _rows_for(self, source: str) -> list[dict]:
        """Run the real `sites_in` over a probe written inside the repo.

        Inside the repo because `sites_in` derives its `rel` from `relative_to(REPO)`; untracked
        because the probe must not move the committed denominator. The file is removed in the
        `finally` so a failure cannot leave a probe behind to be committed by accident.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("_ledger_under_test", LEDGER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        probe = REPO / f"_ledger_spelling_probe_{id(source):x}.py"
        try:
            probe.write_text(source, encoding="utf-8")
            return module.sites_in(probe)
        finally:
            probe.unlink(missing_ok=True)

    def test_every_spelling_of_a_default_read_is_counted(self) -> None:
        for label, source, expected in self.PROBES:
            with self.subTest(spelling=label):
                rows = self._rows_for(source)
                self.assertEqual(
                    len(rows), expected,
                    f"{label}: expected {expected} row(s), got {len(rows)} -- this spelling can "
                    f"add a default read with the denominator unmoved and the gate green.\n"
                    f"rows: {rows}",
                )
                for row in rows:
                    self.assertIn(
                        row["kind"], ("bare-const", "default-spec"),
                        f"{label}: reported as {row['kind']!r}; the kind must be the GLOBAL, not "
                        "the local alias, or two spellings of one site look like two defects.",
                    )

    def test_unrelated_lookalikes_are_not_counted(self) -> None:
        for label, source in self.NEGATIVES:
            with self.subTest(spelling=label):
                self.assertEqual(
                    self._rows_for(source), [],
                    f"{label}: counted as a default read. A matcher that over-matches inflates "
                    "the denominator, which is the same failure as under-matching -- the figure "
                    "stops meaning what it says.",
                )

    def test_the_probes_do_not_disturb_the_committed_denominator(self) -> None:
        """The probe files are untracked and deleted, so N is what it was."""
        before = len(_derive())
        for _, source, _ in self.PROBES:
            self._rows_for(source)
        self.assertEqual(
            before, len(_derive()),
            "running the spelling probes changed the derived count; a probe leaked into the tree.",
        )
        leftover = sorted(p.name for p in REPO.glob("_ledger_spelling_probe_*.py"))
        self.assertEqual(leftover, [], f"probe files left behind: {leftover}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
