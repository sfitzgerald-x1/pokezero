#!/usr/bin/env python3
"""Gates on the gen3 Sleep Talk exclusion sets.

This probe has shipped TWO wrong ground truths. First the sets were hardcoded
while the docstring claimed they came from `data/mods/gen3/moves.ts`. Then they
were parsed out of BASE `data/moves.ts`, which answers 40 -- gen9's number --
because a mod entry's `flags` replaces the parent's wholesale and gen4/gen5 drop
`nosleeptalk` from fly/mimic/sketch/naturepower/struggle.

Nothing tested either. These tests pin the three things that went wrong:
the snapshot matching the resolver, the resolver being asked for GEN3, and a
degraded resolver never being labelled data-backed.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from _showdown_root import showdown_root_str

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "scripts" / "gen3_sleeptalk_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("_stprobe", PROBE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_stprobe"] = module
    spec.loader.exec_module(module)
    return module


def _showdown_root() -> str | None:
    """A BUILT checkout, or None.

    Resolution comes from the shared helper rather than a local candidate list -- a private list
    is how a personal path gets reintroduced, and this file had one. The existence check stays
    local and stricter than the helper's: this probe runs `dist/sim/dex.js`, so a source-only
    checkout that satisfies has_showdown() is not enough here.
    """
    for candidate in (showdown_root_str(), str(REPO / "third_party" / "pokemon-showdown")):
        if candidate and (Path(candidate) / "dist" / "sim" / "dex.js").is_file():
            return candidate
    return None


class Gen3FlagSetTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.mod = _load()
        except Exception as exc:  # native deps unavailable
            self.skipTest(f"probe module not importable: {exc}")

    def test_the_snapshots_still_match_the_resolver(self) -> None:
        """The only thing that catches snapshot ROT.

        The snapshots are load-bearing whenever node or `dist/` is missing. If
        upstream changes gen3's flags and nobody re-derives them, every fallback
        run silently reports last year's answer.
        """
        root = _showdown_root()
        if root is None or shutil.which("node") is None:
            self.skipTest("needs node + a built showdown dist/sim/dex.js")
        charge, nosleeptalk, source = self.mod._move_flag_sets(root)
        self.assertNotIn("SNAPSHOT", source, f"expected a resolver-backed run: {source}")
        self.assertNotIn("UNION", source, f"resolver was degraded: {source}")
        self.assertEqual(charge, self.mod._GEN3_CHARGE_SNAPSHOT)
        self.assertEqual(nosleeptalk, self.mod._GEN3_NOSLEEPTALK_SNAPSHOT)

    def test_the_resolver_is_asked_for_gen3_not_the_base_mod(self) -> None:
        """gen3 is 35, gen9 is 40. Getting 40 means the format lookup missed."""
        root = _showdown_root()
        if root is None or shutil.which("node") is None:
            self.skipTest("needs node + a built showdown dist/sim/dex.js")
        _, nosleeptalk, _ = self.mod._move_flag_sets(root)
        self.assertEqual(len(nosleeptalk), 35, "40 would mean the gen9 table")
        # The five that gen4/gen5 strip. Reporting these as gen3-excluded sends a
        # reader chasing a divergence that does not exist in this format.
        for move in ("fly", "mimic", "naturepower", "sketch", "struggle"):
            self.assertNotIn(move, nosleeptalk, f"{move} is not nosleeptalk in gen3")

    def test_a_format_miss_must_not_silently_return_the_gen9_table(self) -> None:
        """The blocker this test exists for.

        `Dex.forFormat(unknown)` does NOT raise: the missing Format's `.mod` is
        the string "gen9", so `dexes[mod || BASE_MOD]` hands back gen9's table.
        The query must assert what it resolved and fail instead.
        """
        root = _showdown_root()
        if root is None or shutil.which("node") is None:
            self.skipTest("needs node + a built showdown dist/sim/dex.js")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "dist" / "sim").mkdir(parents=True)
        # A dex.js whose format table has been renamed out from under us.
        real = Path(root) / "dist" / "sim" / "dex.js"
        (tmp / "dist" / "sim" / "dex.js").write_text(
            f"const real = require({str(real)!r});\n"
            "exports.Dex = {\n"
            "  formats: {get: () => ({exists: false, mod: 'gen9'})},\n"
            "  forFormat: () => real.Dex.forFormat('gen9randombattle'),\n"
            "};\n",
            encoding="utf-8",
        )
        charge, nosleeptalk, source = self.mod._move_flag_sets(str(tmp))
        self.assertIn("SNAPSHOT", source, f"a format miss must not be data-backed: {source}")
        # Pin THIS guard, not merely the end-to-end outcome. Review showed that
        # asserting only "SNAPSHOT in source" gates the CONJUNCTION of three
        # mutually redundant guards -- remove any two and the test stays green.
        self.assertIn(
            "does not exist in this build", source,
            f"must fall back via the `exists` guard specifically: {source}",
        )
        # And the answer must be gen3's, not the gen9 table it was handed.
        self.assertEqual(nosleeptalk, self.mod._GEN3_NOSLEEPTALK_SNAPSHOT)

    def test_an_existing_format_that_resolves_to_the_wrong_gen_is_refused(self) -> None:
        """The other half of the blocker, previously untested.

        The committed format-miss stub sets `exists: false`, so the JS throws
        before `forFormat` is ever called -- the `dexes[mod || BASE_MOD]` fallback
        the docstring describes was never exercised. This stub reports the format
        as EXISTING and then hands back gen9's dex, which is what a stale
        `formats.js` mapping would do, and pins the gen/currentMod guard.
        """
        root = _showdown_root()
        if root is None or shutil.which("node") is None:
            self.skipTest("needs node + a built showdown dist/sim/dex.js")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "dist" / "sim").mkdir(parents=True)
        real = Path(root) / "dist" / "sim" / "dex.js"
        (tmp / "dist" / "sim" / "dex.js").write_text(
            f"const real = require({str(real)!r});\n"
            "exports.Dex = {\n"
            "  formats: {get: () => ({exists: true, mod: 'gen3'})},\n"
            "  forFormat: () => real.Dex.forFormat('gen9randombattle'),\n"
            "};\n",
            encoding="utf-8",
        )
        charge, nosleeptalk, source = self.mod._move_flag_sets(str(tmp))
        self.assertIn("SNAPSHOT", source, source)
        self.assertIn(
            "expected gen3/gen3", source,
            f"must fall back via the gen/currentMod guard specifically: {source}",
        )
        # 40 would be the gen9 answer leaking through.
        self.assertEqual(len(nosleeptalk), 35, source)

    def test_a_missing_dist_reports_snapshot_not_a_file_path(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        _, _, source = self.mod._move_flag_sets(str(tmp))
        self.assertTrue(source.startswith("SNAPSHOT"), source)

    def test_a_resolver_that_LOSES_a_move_is_suspect_even_at_the_same_size(self) -> None:
        """Cardinality cannot see a substitution.

        Swapping `dive` for a bogus id keeps the count at 35 while silently
        dropping a real gen3 exclusion. The guard is on the SUBSET relation.
        """
        root = _showdown_root()
        if root is None or shutil.which("node") is None:
            self.skipTest("needs node + a built showdown dist/sim/dex.js")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "dist" / "sim").mkdir(parents=True)
        real = Path(root) / "dist" / "sim" / "dex.js"
        (tmp / "dist" / "sim" / "dex.js").write_text(
            f"const real = require({str(real)!r});\n"
            "const base = real.Dex.forFormat('gen3randombattle');\n"
            "exports.Dex = {\n"
            "  formats: {get: () => ({exists: true, mod: 'gen3'})},\n"
            "  forFormat: () => ({\n"
            "    gen: 3, currentMod: 'gen3',\n"
            "    moves: {all: () => [...base.moves.all()].map(m =>\n"
            "      m.id === 'dive' ? {id: 'zzbogus', flags: m.flags} : m)},\n"
            "  }),\n"
            "};\n",
            encoding="utf-8",
        )
        charge, nosleeptalk, source = self.mod._move_flag_sets(str(tmp))
        self.assertIn("SUSPECT", source, f"a lost move must be flagged: {source}")
        self.assertIn("dive", source)
        # The union keeps the real move so downstream cannot lose an exclusion.
        self.assertIn("dive", charge)


if __name__ == "__main__":
    unittest.main()
