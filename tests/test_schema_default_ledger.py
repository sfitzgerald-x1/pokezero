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
import ast
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# tests/data, not corpus/: corpus/ is gitignored, so an allowlist there is invisible to
# everyone else and the gate silently cannot run. Caught by `git add` refusing the path.
ALLOWLIST = REPO / "tests" / "data" / "schema_default_allowlist.json"
# The most rows the allowlist has ever legitimately held. Only ever lowered.
HIGH_WATER_MARK = 249
LEDGER = REPO / "scripts" / "schema_default_ledger.py"


def _constructor_names() -> tuple[str, ...]:
    """The ledger's own CONSTRUCTOR_NAMES, read rather than mirrored.

    A literal copy in this file is what left `__post_init__` unpinned: narrowing the gate's tuple to
    drop only that name stayed green, while the comment claimed the assertion was derived from it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ledger_ctor_names", LEDGER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return tuple(module.CONSTRUCTOR_NAMES)


def _sweep_probe_residue() -> list[str]:
    """Remove any probe file a crashed or interrupted run left in the live src/ tree.

    Module-level and called by every reader, not just the writer. Review reproduced the failure and
    so did I: a run killed mid-probe leaves `_surface_probe_*.py` behind, the next run derives an
    extra surface from it, and the DOCSTRING test fails -- pointing at the docstring, which is fine.
    A test whose failure names the wrong file is worse than a slow one.
    """
    swept = []
    for stale in (REPO / "src" / "pokezero").glob("_surface_probe_*.py"):
        swept.append(stale.name)
        stale.unlink(missing_ok=True)
    for stale in (REPO / "src" / "pokezero").rglob("_ledger_spelling_probe_*.py"):
        swept.append(stale.name)
        stale.unlink(missing_ok=True)
    return swept


def _derive() -> list[dict]:
    """Re-derive from the tree. Never read a cached count -- that is the error being retired."""
    _sweep_probe_residue()
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

    def _rendered_kinds_table(self) -> str:
        """The kinds table AS THE TOOL RENDERS IT. The docstring must contain this verbatim."""
        proc = subprocess.run(
            [sys.executable, str(LEDGER), "--render-kinds-table"], capture_output=True, text=True
        )
        if proc.returncode == 2:
            raise unittest.SkipTest(
                "the ledger reported UNPARSED file(s), so the counts are measured against an "
                "incomplete denominator and comparing them to the docstring proves nothing."
            )
        self.assertEqual(
            proc.returncode, 0,
            f"`--render-kinds-table` exited {proc.returncode}.\n{proc.stderr[-2000:]}",
        )
        rendered = proc.stdout.rstrip("\n")
        # NON-VACUITY, tied to a measurement rather than to a constant. `render_kinds_table` returns
        # '' when the tool emits no non-surface kinds, and an empty docstring region would then
        # compare equal to it -- assertEqual('', '') passes, which is the exact vacuity this file
        # exists to prevent. The empty case is ALSO the migration's goal state (bare-const and
        # default-spec burned down to 0), so it cannot simply be forbidden; it has to be
        # DISTINGUISHED from a tool that produced nothing for any other reason.
        rows = _derive()
        kinds = sorted(
            {r["kind"] for r in rows if not r["kind"].startswith("implicit:") and r["kind"] != "UNPARSED"}
        )
        lines = [l for l in rendered.split("\n") if l.strip()]
        self.assertEqual(
            len(lines), len(kinds),
            f"the rendering has {len(lines)} row(s) but the tool emits {len(kinds)} non-surface "
            f"kind(s) {kinds}. An empty or short rendering compared against an empty or short "
            "docstring region would pass while checking nothing.",
        )
        return rendered

    def _docstring_kinds_region(self) -> tuple[str, int, int]:
        """(region, start, end) -- the docstring span the tool owns, blank lines dropped in `region`.

        This is the region the tool owns. Extracting a REGION rather than matching rows means an
        extra row, a duplicated row, a stale row, or a reformatted row all land inside it and all
        show up as a diff.

        The SPAN is returned alongside the text because the caller needs to look at everything
        outside it. Computing "outside" as `doc.replace(region, "", 1)` was wrong: `region` has had
        its blank lines dropped, so it is not necessarily a literal substring of the docstring, and a
        single blank line inserted between two generated rows made the replace a no-op. Every kind
        name then appeared "outside" and the test failed with a message naming the wrong cause --
        the misdiagnosis class this file polices elsewhere. Slicing by index cannot do that.
        """
        doc = self._docstring()
        try:
            start = doc.index("is any of:")
            start = doc.index("\n", start) + 1
            end = doc.index("  implicit:", start)
        except ValueError:
            self.fail(
                "the ledger docstring no longer has an `is any of:` ... `  implicit:` kinds table, "
                "so this guard has nothing to compare. An empty region would make the comparison "
                "vacuous, which is the failure mode this whole test exists to prevent."
            )
        return "\n".join(l for l in doc[start:end].split("\n") if l.strip()), start, end

    def _docstring(self) -> str:
        return ast.get_docstring(ast.parse(LEDGER.read_text(encoding="utf-8"))) or ""

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

        _sweep_probe_residue()

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

    def test_the_docstring_kinds_table_is_byte_equal_to_the_tool_rendering(self) -> None:
        """The kinds table is GENERATED. Compare it byte-for-byte; do not parse it.

        This replaces five rounds of narrowing a regex that tried to RECOGNISE a hand-written
        table. Each round closed the escapes the previous reviewer named and left the next ones
        open: first two reformats, then seventeen more -- an uppercase or capitalised kind name, an
        underscore, a `*` or `-` bullet the table already uses elsewhere, a trailing colon in the
        house style of the very next row, `[16]`, `n=16`, `(16 rows)` which this docstring itself
        uses nine lines below, the count moved to the following line. Plus a containment check that
        masked an unreadable row whenever its text was a substring of a readable one, and
        last-write-wins on a duplicated row, so a WRONG count inserted above the right one was
        discarded in silence.

        A grammar cannot win that. Every version was one character from the next escape, and each
        fix arrived with a claim that the property now held. So there is no grammar: the tool emits
        the block, and the docstring region must equal it exactly. A reformat is a diff rather than
        a disappearance. A duplicate row is a diff. A retired kind is a diff. A wrong count is a
        diff. Regenerate with:

            python scripts/schema_default_ledger.py --render-kinds-table
        """
        region, span_start, span_end = self._docstring_kinds_region()
        self.assertEqual(
            region, self._rendered_kinds_table(),
            "the ledger docstring's kinds table is not byte-equal to what the tool renders.\n"
            "Regenerate it:  python scripts/schema_default_ledger.py --render-kinds-table\n"
            "This table is GENERATED, not hand-maintained -- any difference (a changed count, an "
            "added or retired kind, a duplicated row, a reformatted row, stray prose inside the "
            "region) is a real difference and none of them can hide from a byte comparison.",
        )
        # Byte-equality closes everything INSIDE the region and says nothing about outside it. That
        # gap is not theoretical: retire a kind in the tool and MOVE its row above `is any of:` or
        # below the `implicit:` block, and the stale count survives with this test green -- the
        # "a stale row is stale wherever it sits" property failing again, now on region bounds
        # instead of on grammar. Measured, both escapes were green before this check.
        #
        # Closed using the tool's OWN vocabulary as the source of truth: `KIND_DESCRIPTIONS` names
        # every kind the ledger knows, including one whose count has reached zero, so a retired
        # kind's name is still known here even though it no longer appears in the rows. Each such
        # name must occur in the docstring ONLY inside the generated region.
        doc = self._docstring()
        outside = doc[:span_start] + doc[span_end:]  # by INDEX; see _docstring_kinds_region
        stray = sorted(k for k in self._kind_vocabulary() if k in outside)
        self.assertEqual(
            stray, [],
            f"kind name(s) {stray} appear in the ledger docstring OUTSIDE the generated kinds "
            "table. Only the generated region may name a kind.\n"
            "TWO WAYS TO HIT THIS, and the message cannot tell them apart, so both are listed:\n"
            "  1. a retired kind's ROW was moved out of the region -- that is how a stale count "
            "survives the byte comparison, and it is what this check exists for;\n"
            "  2. ordinary PROSE mentions a kind by name. Also disallowed, and deliberately: prose "
            "naming a kind beside a number is how every stale figure in this docstring got there. "
            "Rewrite it without the name.\n"
            "Naming both cases is the point -- an earlier version of this message asserted (1) only, "
            "and sent a maintainer who had written prose looking for a row they had never moved.\n"
            "KNOWN LIMIT, stated rather than implied: this uses the tool's KIND_DESCRIPTIONS as the "
            "vocabulary, so a row whose kind has been deleted from KIND_DESCRIPTIONS *and* retired "
            "from the emitted rows *and* moved out of the region would still escape. That is three "
            "deliberate edits, and the first of them is the one to review.",
        )
        # THE SUMMARY WORD, which was unpinned prose. The docstring says "All EIGHT counts above are
        # held" -- six surface rows plus the kind rows. Retire a kind, remove its row correctly and
        # regenerate, and the sentence claims eight while seven exist: this docstring's own documented
        # staling pattern, one level up from the rows it now pins. Derived from both halves so it
        # tracks either changing.
        surfaces = len(self._surfaces())
        n_kinds = len([l for l in region.split("\n") if l.strip()])
        total = surfaces + n_kinds
        words = {
            0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX",
            7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN", 11: "ELEVEN", 12: "TWELVE",
        }
        self.assertIn(
            total, words,
            f"{total} held counts, which this guard has no spelled word for. Add it -- do NOT fall "
            "back to the digit, which is how a sibling guard in this file went inert.",
        )
        # The SET of counts stated, not "the right one appears somewhere". `assertIn` over the whole
        # docstring detects a MISSING total and not a WRONG one: state EIGHT in the real sentence and
        # plant "All NINE counts" in prose, and it passes. That is the identical weakness the surface
        # guard in this file already had and had fixed, reintroduced one guard over -- so a wrong
        # total is now caught wherever it sits, in exactly the form the right one takes.
        # Case-insensitive, compared uppercased, and INTERSECTED WITH THIS GUARD'S OWN NUMBER
        # VOCABULARY. `[A-Z]+` missed a wrong total written "All Nine counts", so it was widened to
        # `[A-Za-z]+` -- which then swallowed any English word, so ordinary prose ("all the counts
        # below come from one run", "All surface counts are parsed") was read as a STATED TOTAL and
        # failed with a message telling the author they had stated a wrong total. They had stated no
        # number at all. That is the misdiagnosis class, introduced by the commit whose purpose was
        # to remove a misdiagnosing message -- and the sibling surface guard sixty lines below
        # documents refusing exactly this broadening, which I did not read before widening.
        #
        # Intersecting with `words.values()` drops the false positives and keeps every kill EXCEPT
        # one, named below: a wrong number word inside this guard's 0-12 vocabulary is still caught
        # wherever it sits, and a sentence containing no number word is not a total at all.
        #
        # STATED NARROWNESS, because the sibling guard states its own and this one did not. Only
        # `all <NUMBER-WORD-IN-0..12> counts` is recognised. These slip past while the correct total
        # is also present:
        #   "All of the NINE counts"      intervening words before the number
        #   "NINE counts in total"        no leading `all`
        #   "All TWENTY-NINE counts"      the hyphen breaks `[A-Za-z]+`
        #   "All 9 counts"                digits, not a word
        #   "All NINE derived counts"     any word between the number and `counts`
        #   "All THIRTEEN counts"         a number word OUTSIDE this guard's 0-12 `words` map -- the
        #       one case the intersection LOST relative to the un-intersected form, which caught it.
        #       Kept anyway: `assertIn(total, words)` above hard-fails if the real total ever exceeds
        #       12, so the uncaught case is a maintainer writing >12 when the truth is <=12 -- a gross
        #       typo, not the staling drift this guard exists for. The alternative was a guard that
        #       fails on ordinary English prose, which is strictly worse.
        # The first five are deliberate: catching them means matching prose, which is the defect
        # above. A total absent entirely still fails, since the empty set is not `{words[total]}`.
        stated_totals = {
            m.upper() for m in re.findall(r"(?i)all ([A-Za-z]+) counts", doc)
        } & set(words.values())
        self.assertEqual(
            stated_totals, {words[total]},
            f"the docstring must state 'All {words[total]} counts' and no other total -- {surfaces} "
            f"surface row(s) plus {n_kinds} kind row(s) -- and it states "
            f"{sorted(stated_totals) or 'none'}. It is the one figure in this table that describes "
            "the table itself, and nothing held it until now.",
        )
        # NON-VACUITY, the case my own round-11 message overstated. `len(lines) == len(kinds)` binds a
        # SHORT rendering; it does NOT bind empty-vs-empty, because 0 == 0. And an empty kinds table
        # IS the migration's goal state, so it cannot simply be forbidden -- it has to be
        # ACKNOWLEDGED, in the docstring, by a human. Round 11 printed a note instead, and pytest
        # swallows stdout for passing tests, so the "announcement" was invisible exactly where it ran.
        if n_kinds == 0:
            self.assertIn(
                "GOAL STATE REACHED", doc,
                "the ledger emits no non-surface kinds -- every kind has burned down to zero, which "
                "is this migration's goal state. Both sides of the byte comparison are then empty "
                "and it proves nothing, so the docstring must say 'GOAL STATE REACHED' to record "
                "that a human saw it. Without that, an empty table and a broken tool are "
                "indistinguishable.\n"
                "WRITE THE ACKNOWLEDGEMENT WITHOUT NAMING THE KINDS. Kind names may appear only "
                "inside the generated table, and the stray-name check above will reject prose that "
                "spells one -- an earlier version of this very message told you to write "
                "'<kind> and <kind> have burned down to zero', which then failed for a reason that "
                "had nothing to do with what you wrote.",
            )

    def _kind_vocabulary(self) -> list[str]:
        """Every non-surface kind the ledger KNOWS, from its own KIND_DESCRIPTIONS."""
        import importlib.util

        _sweep_probe_residue()
        spec = importlib.util.spec_from_file_location("_ledger_kinds", LEDGER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        vocab = sorted(module.KIND_DESCRIPTIONS)
        self.assertTrue(
            vocab,
            "the ledger has an empty KIND_DESCRIPTIONS, so this guard has no vocabulary and the "
            "stray-row check below would pass over anything.",
        )
        return vocab

    def test_the_docstring_states_the_surface_count_it_actually_has(self) -> None:
        doc = LEDGER.read_text(encoding="utf-8")
        n = len(self._surfaces())
        # Spelled words only, and an unmapped count is a HARD FAILURE rather than a fallback to
        # the digit. This guard was inert exactly once: the map covered 7/8/9, the count became
        # SIX, `words.get(n, str(n))` degraded the assertion to `assertIn("6", docstring)`, and the
        # only "6" in the docstring is the one inside `bare-const (16)` on an unrelated line. The
        # docstring could have said EIGHT, or named no count at all, and this test stayed green
        # while claiming to hold it. A digit is a substring of other digits; a word is not, which
        # is the whole reason this guard spells the number.
        words = {0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE",
                 6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN"}
        if n not in words:
            self.fail(
                f"{n} surfaces, which this guard has no spelled word for. Add it to `words` -- do "
                "NOT fall back to the digit, which is how this assertion went inert before."
            )
        # ANCHORED on the table sentence, not searched for anywhere in the docstring. `assertIn`
        # over the whole text was the second inert version of this guard: the word only had to
        # appear SOMEWHERE, and this docstring says EIGHT twice in prose ("WAS EIGHT", "All EIGHT
        # counts above"). So a regression from six surfaces back to eight -- the exact prior value
        # this docstring narrates -- passed with the table still reading "All SIX". Spelling the
        # number fixed digit-substring matching and left prose-substring matching wide open: a word
        # is not a substring of other numbers, but it is a substring of sentences containing it.
        # SCOPED to the `implicit:` entry's own paragraph, not searched across the whole docstring.
        # A phrase match over the full text was still defeatable: set the table to "All NINE" and
        # plant "All SIX\n surfaces" in the WAS-EIGHT paragraph, and it passed. Scoping means prose
        # elsewhere is not merely unlikely to collide, it is out of range.
        #
        # `\s+` rather than `\s*\n\s*`: requiring a line break between the word and "surfaces" made
        # the guard fail on legitimate rewrapping, reporting that the table "does not say so" when
        # it plainly did. A guard that fires on reflow gets edited out rather than obeyed.
        body = doc.split('"""')[1]
        try:
            para = body[body.index("implicit:"):].split("\n\n")[0]
        except ValueError:  # pragma: no cover - guarded so the scope cannot silently empty
            self.fail(
                "the ledger docstring no longer has an `implicit:` entry, so there is no surface "
                "table paragraph to hold the count. An empty scope makes this assertion vacuous."
            )
        # Collect every count the paragraph states in the form `All <UPPERCASE-WORD> surfaces` and
        # require the set to be exactly the right one. Asserting only that the correct phrase appears
        # detects a MISSING count but not a WRONG one: with the table set to "All NINE surfaces" and
        # "All SIX surfaces" appended lower in the same paragraph, the right phrase was present, so
        # it passed while the table misstated the count. Deleting the blank line that terminates the
        # paragraph widened the scope and did the same. A set comparison fails on the extra "NINE"
        # regardless of where either phrase sits.
        #
        # SCOPE OF THIS GUARD, stated because the obvious reading is broader than the truth. It binds
        # on POSITION and is narrow on SPELLING: only `All <UPPERCASE-WORD> surfaces` is recognised,
        # so a wrong count written as "All nine surfaces", "All 9 surfaces", "All TWENTY-NINE
        # surfaces", or -- the one worth knowing -- "All NINE derived surfaces", with any word
        # between, is invisible while the correct phrase is also present. The intervening-adjective
        # case is the same wording family as the original defect ("seven derived surfaces") that this
        # guard's failure message cites. Broadening the pattern to catch them would start matching
        # ordinary prose about surfaces, so the narrowness is deliberate rather than overlooked.
        #
        # The earlier "exactly once" requirement is deliberately GONE: a set comparison cannot see
        # duplicates, so two correct occurrences now pass where they used to fail. Restating the
        # right count is harmless; stating a wrong one is not, and that is what this now catches.
        stated = set(re.findall(r"All ([A-Z]+)\s+surfaces\b", para))
        self.assertEqual(
            stated, {words[n]},
            f"there are {n} surfaces, so the docstring's surface-table paragraph must state "
            f"'All {words[n]} surfaces' and no other count; it states {sorted(stated) or 'none'}.\n"
            f"  paragraph searched:\n{para}\n"
            "The original text said 'seven derived surfaces' while listing an alternate "
            "constructor as derived and omitting a derived surface with no open sites.",
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


class SurfaceDerivationSeesEverySpellingTest(unittest.TestCase):
    """Every way to DECLARE a defaulted field must derive its surface.

    This matters more than `sites_in`'s spelling coverage, and for six rounds it had none. A missed
    read loses one row; a missed DECLARATION loses the surface and with it EVERY call site --
    `LocalShowdownConfig` alone is 133 of the 390. So an under-derived surface is the single largest
    way this denominator can be wrong, and `derive_surfaces` recognised exactly one spelling while
    `sites_in` accumulated four. The round-5 alias fix was applied to `sites_in` and never here.

    Review quantified it end to end: with the same defaulted field and ten identical constructions,
    the plain annotated spelling moved N by +11 (1 declaration + 10 callers) while
    `field(default=GLOBAL)`, an aliased global, and an un-annotated attribute each moved it by +1 --
    the ten callers invisible. Combined with a relative import even the +1 vanished.
    """

    PROBES = [
        ("annotated default (the only spelling derived before round 7)",
         "from pokezero.observation import OBSERVATION_SCHEMA_VERSION\n"
         "class Surf:\n    v: str = OBSERVATION_SCHEMA_VERSION\n"),
        ("UN-annotated class attribute (ast.Assign, not AnnAssign)",
         "from pokezero.observation import OBSERVATION_SCHEMA_VERSION\n"
         "class Surf:\n    v = OBSERVATION_SCHEMA_VERSION\n"),
        ("aliased global -- the round-5 fix, never applied to this function",
         "from pokezero.observation import OBSERVATION_SCHEMA_VERSION as SV\n"
         "class Surf:\n    v: str = SV\n"),
        ("dataclasses field(default=...) -- 201 `field(default` uses in src/",
         "from dataclasses import field\n"
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "class Surf:\n    v: object = field(default=DEFAULT_REPLAY_OBSERVATION_SPEC)\n"),
        ("field(default_factory=lambda: ...)",
         "from dataclasses import field\n"
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "class Surf:\n"
         "    v: object = field(default_factory=lambda: DEFAULT_REPLAY_OBSERVATION_SPEC)\n"),
        ("module-qualified, one level",
         "from pokezero import observation\n"
         "class Surf:\n    v: str = observation.OBSERVATION_SCHEMA_VERSION\n"),
        ("module-qualified, DOTTED base -- the one-level bug, left in this function for a round",
         "import pokezero.observation\n"
         "class Surf:\n"
         "    v: str = pokezero.observation.OBSERVATION_SCHEMA_VERSION\n"),
        ("the VALUE side, dotted base",
         "import pokezero.showdown\n"
         "class Surf:\n"
         "    v: int = pokezero.showdown.DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count\n"),
        ("a function parameter default, not a class field",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "def surf_fn(spec=DEFAULT_REPLAY_OBSERVATION_SPEC):\n    return spec\n"),
        ("a CONSTRUCTOR parameter default -- must key on the CLASS, not on __init__",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "class Surf:\n"
         "    def __init__(self, spec=DEFAULT_REPLAY_OBSERVATION_SPEC):\n"
         "        self.spec = spec\n"),
        ("__new__, same rule",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "class Surf:\n"
         "    def __new__(cls, spec=DEFAULT_REPLAY_OBSERVATION_SPEC):\n"
         "        return super().__new__(cls)\n"),
        ("__post_init__, the third name in the gate's tuple -- previously unpinned",
         "from dataclasses import dataclass\n"
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "class Surf:\n"
         "    def __post_init__(self, spec=DEFAULT_REPLAY_OBSERVATION_SPEC):\n"
         "        self.spec = spec\n"),
        ("a keyword-only parameter default",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "def surf_fn(*, spec=DEFAULT_REPLAY_OBSERVATION_SPEC):\n    return spec\n"),
        # POSITIONAL-ONLY. `ast.arguments.defaults` covers posonlyargs + args COMBINED, so slicing
        # `a.args` alone misaligned the pairing. Round 8's finding, and the only escape in this file
        # that mis-ANSWERS rather than staying silent.
        ("positional-only after a defaulted positional",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "def surf_fn(a=1, /, spec=DEFAULT_REPLAY_OBSERVATION_SPEC):\n    return spec\n"),
        ("the global itself positional-only, with a plain kwarg after it",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "def surf_fn(spec=DEFAULT_REPLAY_OBSERVATION_SPEC, /, other=1):\n    return spec\n"),
        ("positional-only, nothing after it",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "def surf_fn(spec=DEFAULT_REPLAY_OBSERVATION_SPEC, /):\n    return spec\n"),
        # B4 spellings.
        ("field(default_factory) wrapping the global in an expression",
         "from dataclasses import field, replace\n"
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "class Surf:\n"
         "    v: object = field(\n"
         "        default_factory=lambda: replace(DEFAULT_REPLAY_OBSERVATION_SPEC)\n    )\n"),
        ("a class attribute inside a conditional block in the class body",
         "import sys\n"
         "from pokezero.observation import OBSERVATION_SCHEMA_VERSION\n"
         "class Surf:\n"
         "    if sys.version_info >= (3, 11):\n"
         "        v: str = OBSERVATION_SCHEMA_VERSION\n"),
    ]

    NEGATIVES = [
        ("a per-version constant, deliberately not a surface",
         "from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V4\n"
         "class Surf:\n    v: str = OBSERVATION_SCHEMA_VERSION_V4\n"),
        ("an unrelated default",
         "class Surf:\n    v: int = 7\n"),
        ("field() with no default at all",
         "from dataclasses import field\n"
         "class Surf:\n    v: object = field(init=False)\n"),
        # A module-qualified LOOKALIKE on an unrelated package. The declaration side had no
        # module-root resolution at all, so this derived a surface -- and a spurious surface costs
        # EVERY call site of that class name, the more expensive over-match direction.
        ("a lookalike rooted outside pokezero",
         "import numpy\n"
         "class Surf:\n    v: str = numpy.OBSERVATION_SCHEMA_VERSION\n"),
        # NOT tested as a negative, deliberately, and stated rather than dropped:
        # `pokezero.not_a_module.OBSERVATION_SCHEMA_VERSION` DOES derive a surface. Resolution here
        # is by chain ROOT, so anything rooted at a real pokezero module passes. Making it strict
        # would mean resolving every segment of an arbitrary chain against the filesystem, and the
        # payoff is only for code that raises AttributeError at import and therefore cannot reach
        # the default at all. Over-matching unimportable code inflates nothing that runs. The
        # blocker this negative set exists for -- a lookalike rooted OUTSIDE pokezero -- is above.
        # A LOCAL VARIABLE in a method is not a field. Walking the whole ClassDef derived a bogus
        # `default_spec` kwarg from `from_dict`'s local, moving N 390 -> 398 and rewriting the
        # `unclosed` field of 101 rows -- which in a diff would have read as "eight new sites".
        # A LAMBDA parameter default. Round 8 added `ast.Lambda` to the walked node types and
        # claimed the spelling was "fixed and pinned"; both halves were false. `ast.Lambda` has no
        # `.name`, so this raised AttributeError -- and since SURFACES is built at import, ONE such
        # lambda anywhere in src/ killed the ledger in every mode and turned the gate into "ledger
        # derivation failed". This negative is the pin that was missing: an anonymous callable has no
        # call-site name to match, so deriving nothing is correct, and the ledger must still RUN.
        ("a lambda parameter default -- derives nothing, and must not crash",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "f = lambda spec=DEFAULT_REPLAY_OBSERVATION_SPEC: spec\n"),
        ("a local variable inside a method, not a field",
         "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "class Surf:\n"
         "    def build(self):\n"
         "        default_spec = DEFAULT_REPLAY_OBSERVATION_SPEC\n"
         "        return default_spec\n"),
    ]

    def _surfaces_with(self, source: str) -> dict:
        """Re-derive SURFACES with a probe module dropped into src/pokezero/.

        Inside `src/` because `derive_surfaces()` only scans there -- which is itself one of the
        documented open routes. Untracked and deleted in the `finally`, so the committed
        denominator cannot move.
        """
        import importlib.util

        # Sweep any residue from a crashed or concurrent earlier run FIRST. A leaked probe in the
        # live src/ tree reddens three unrelated tests, which review reproduced -- the failure then
        # points at the wrong thing entirely.
        _sweep_probe_residue()
        probe = REPO / "src" / "pokezero" / f"_surface_probe_{id(source):x}.py"
        try:
            probe.write_text(source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(f"_ledger_surf_{id(source):x}", LEDGER)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return dict(module.SURFACES)
        finally:
            probe.unlink(missing_ok=True)

    def test_every_declaration_spelling_derives_its_surface(self) -> None:
        for label, source in self.PROBES:
            with self.subTest(spelling=label):
                surfaces = self._surfaces_with(source)
                name = "Surf" if "class Surf" in source else "surf_fn"
                # READ from the ledger module's own tuple, not a literal mirroring it. The previous
                # version claimed "derived from the skip list" while hardcoding
                # ("__init__", "__new__", "__post_init__") -- a false statement about the test's own
                # wiring, in the same category as the false docstring that blocked round 10, and it
                # left `__post_init__` unpinned: narrowing the gate's tuple to drop only that name
                # stayed green. Reading the tuple means the assertion cannot drift from the gate.
                for dunder in sorted(_constructor_names()):
                    self.assertNotIn(
                        dunder, surfaces,
                        f"a phantom `{dunder}` surface was derived alongside the class. Keying a "
                        f"constructor on its own name makes every `Surf(...)` call site invisible "
                        f"AND scores rows on any literal `x.{dunder}(...)`.",
                    )
                self.assertIn(
                    name, surfaces,
                    f"{label}: the surface was NOT derived, so every call site of it is invisible "
                    f"to the ledger. Derived surfaces: {sorted(surfaces)}",
                )
                # A CONSTRUCTOR probe declares `class Surf` and defaults `spec`, so keying the
                # expectation on "is there a class" alone gave the wrong answer for it.
                # The constructor list again read rather than mirrored -- the first version of this
                # expectation named `__init__` and `__new__` only, so adding a `__post_init__` probe
                # failed against correct code. Same defect as the assertion below, one line apart.
                is_ctor = any(f"def {d}" in source for d in _constructor_names())
                expected_field = {"spec"} if (is_ctor or name == "surf_fn") else {"v"}
                self.assertEqual(
                    surfaces[name], expected_field,
                    f"{label}: derived the surface but named {sorted(surfaces[name])} as the open "
                    "route instead of the field that actually defaults. Naming the WRONG kwarg is "
                    "worse than naming none: a call site scores CLOSED as soon as it passes that "
                    "kwarg, which closes nothing -- and a positional-only parameter can never be "
                    "closed by keyword at all.",
                )

    def test_unrelated_declarations_do_not_derive_a_surface(self) -> None:
        for label, source in self.NEGATIVES:
            with self.subTest(spelling=label):
                surfaces = self._surfaces_with(source)
                self.assertNotIn(
                    "Surf", surfaces,
                    f"{label}: derived a surface it should not have. Over-derivation inflates N "
                    "through every call site of a class that does not default to the global.",
                )

    def test_the_probes_do_not_disturb_the_committed_surface_set(self) -> None:
        before = self._surfaces_with("x = 1\n")
        for _, source in self.PROBES:
            self._surfaces_with(source)
        self.assertEqual(
            sorted(before), sorted(self._surfaces_with("x = 1\n")),
            "the probes changed the derived surface set; one leaked into src/.",
        )
        leftover = sorted(p.name for p in (REPO / "src" / "pokezero").glob("_surface_probe_*.py"))
        self.assertEqual(leftover, [], f"probe files left behind: {leftover}")


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
        # A REAL module outside the old hand-listed ("observation", "showdown") pair. An earlier
        # revision of this probe named `replay`, which does not exist in src/pokezero/ -- so once
        # the matcher started resolving module names against the filesystem the probe correctly
        # scored 0 and the test caught its own fiction. A probe naming a module that cannot be
        # imported proves nothing about a spelling.
        ("`from pokezero import <mod>` outside the old hand-listed pair",
         "from pokezero import local_showdown\n"
         "def f():\n    return local_showdown.OBSERVATION_SCHEMA_VERSION\n", 1),
        ("RELATIVE `from . import <mod>`  (round-7 escape; live in the package)",
         "from . import observation\n"
         "def f():\n    return observation.OBSERVATION_SCHEMA_VERSION\n", 1),
        ("relative and aliased",
         "from . import observation as O\n"
         "def f():\n    return O.OBSERVATION_SCHEMA_VERSION\n", 1),
        # Written into the real `mcts_eval` subpackage, because `..` only means `src/pokezero`
        # from INSIDE a subpackage -- from `src/pokezero/` itself it means `src/`, where no
        # `showdown.py` exists. The tree's deepest relative import really is level 2
        # (src/pokezero/mcts_eval/lattice.py:31), which is what this probe imitates.
        ("relative, one level up  (from inside a subpackage)",
         "from .. import showdown as S\n"
         "def f():\n    return S.DEFAULT_REPLAY_OBSERVATION_SPEC\n", 1, "mcts_eval"),
        # `from .<module> import <name>` -- level > 0 WITH a non-None node.module. This is the ONLY
        # form the round-9 resolution fix changes, and no probe used it, so reverting that fix left
        # the entire gate green: N stayed 390, the allowlist stayed byte-identical, and all three
        # surface tests passed. Moving the other relative probes into src/pokezero/ made them
        # SURVIVE the fix; it did not make them BIND it. This file's own docstring diagnoses that
        # exact failure -- "each fix was verified once by hand and pinned by no test".
        ("relative import from a SUBPACKAGE  (`from .mcts_eval import lattice`)",
         "from .mcts_eval import lattice\n"
         "def f():\n    return lattice.OBSERVATION_SCHEMA_VERSION\n", 1),
        ("relative import of the global itself, aliased",
         "from .observation import OBSERVATION_SCHEMA_VERSION as SV\n"
         "def f():\n    return SV\n", 1),
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
        # A POKEZERO-rooted lookalike. Every other negative is rooted outside the package (numpy),
        # so nothing guarded the over-match direction for a name imported FROM pokezero: the
        # ImportFrom branch adds every imported name to module_roots, classes included, so
        # `ObservationSpec.OBSERVATION_SCHEMA_VERSION` scored a row. Zero live occurrences, but
        # this file calls over-matching "the same failure as under-matching" and must test it.
        ("a class imported from pokezero, not a module",
         "from pokezero.observation import ObservationSpec\n"
         "def f():\n    return ObservationSpec.OBSERVATION_SCHEMA_VERSION\n"),
        # The negative half of the same form: `observation` is a NAME re-exported by the subpackage,
        # not a submodule of it, so it must not become a module base. Before the round-9 fix this
        # scored 1; the hardcoded `src/pokezero` root made it look like a module.
        ("a NAME imported from a subpackage, not a module",
         "from .mcts_eval import observation\n"
         "def f():\n    return observation.OBSERVATION_SCHEMA_VERSION\n"),
        ("a module that does not exist in the tree",
         "from pokezero import not_a_real_module\n"
         "def f():\n    return not_a_real_module.OBSERVATION_SCHEMA_VERSION\n"),
        # A WRITE is not a read. Without a ctx=Load gate the assignment target at
        # `showdown.py:1143` scored a row, so N was 391 where the truth is 390 -- and that
        # contradicted the docstring ("a read of"), DEFINITION_SITES's own rationale, and this
        # file's insistence that over-matching is the same failure as under-matching.
        ("an assignment TARGET, not a read",
         "OBSERVATION_SCHEMA_VERSION = 'x'\n"),
        ("an annotation with no value",
         "OBSERVATION_SCHEMA_VERSION: str\n"),
    ]

    def _rows_for(self, source: str, subdir: str = "") -> list[dict]:
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
        # INSIDE src/pokezero/, not at the repo root. A relative import only means anything
        # from inside the package: once `is_pokezero_submodule` began resolving
        # `from . import X` against the IMPORTING file's own directory (round 9), a probe at
        # the repo root correctly resolved nothing and the three relative-import probes went
        # red. A probe has to live where the code it imitates lives, or it tests a different
        # question than the one it claims to.
        probe = (
            REPO / "src" / "pokezero" / subdir / f"_ledger_spelling_probe_{id(source):x}.py"
        )
        try:
            probe.write_text(source, encoding="utf-8")
            return module.sites_in(probe)
        finally:
            probe.unlink(missing_ok=True)

    def test_every_spelling_of_a_default_read_is_counted(self) -> None:
        for probe in self.PROBES:
            label, source, expected = probe[0], probe[1], probe[2]
            subdir = probe[3] if len(probe) > 3 else ""
            with self.subTest(spelling=label):
                rows = self._rows_for(source, subdir)
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
        for probe in self.PROBES:
            self._rows_for(probe[1], probe[3] if len(probe) > 3 else "")
        self.assertEqual(
            before, len(_derive()),
            "running the spelling probes changed the derived count; a probe leaked into the tree.",
        )
        leftover = sorted(
            p.name
            for p in (REPO / "src" / "pokezero").rglob("_ledger_spelling_probe_*.py")
        )
        self.assertEqual(leftover, [], f"probe files left behind: {leftover}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
