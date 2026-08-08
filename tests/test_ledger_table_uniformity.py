"""Re-derive the known-gaps ledger's table geometry from a RENDERER'S rule, not a grep.

WHY THIS MODULE EXISTS. C116 Phase 4 item 14's standing instruction is that *"no entry
joins the known-gaps ledger without a pool-reachability check recorded next to it"*
(`reports/c124_a6_is_knowable.md` §2). `reports/c138_known_gaps_ledger.md` records that
check in a dedicated **Reachability evidence** column -- and a table row carrying more cell
delimiters than its header does not merely look wrong in GFM, it **silently drops the
overflow**. So a bare `|` anywhere in a row's prose can delete that row's reachability
check from the rendered document while leaving it in the bytes: the ledger passes
inspection by grep and fails item 14 in the only form a reader sees.

THAT HAS HAPPENED, AND THE INSTANCE IS VERIFIED, not inferred. At `a587e614^` -- #1166's
tree, before its own fix -- `G37` carried 20 pipes and `G37b` 9, against a 5-column
(6-pipe) header. Rendered through `cmarkgfm` those rows come back with **`Reachability
evidence` EMPTY**, `Class` reading `cant`, and `Observed` reading `Attractor` / a fragment
of a source path. Both rows' actual pool checks and Observed values were gone from the
document. #1166 fixed them at `a587e614`, and the whole file has been clean since:
re-derived at `f876803e` (0 over-delimited rows), `a587e614^` (**2**), `a587e614` (0) and
`553cf2c3` (0).

So `reports/c146_negative_claim_audit.md`'s *"all 9 tables now have uniform column
counts"* was **TRUE when written**, and item 14 is **met** at `553cf2c3`. What #1166 did
not leave behind was a control. Its own disposition says so: *"Not pinned: a markdown
table-integrity check does not belong in a counter census ... Filed as a follow-up
instead."* This module is that follow-up. It exists to stop the NEXT G37, not to repair a
live breach -- there is none.

⚠ AND IT NEARLY BECAME AN INSTANCE OF THE DEFECT IT GUARDS. The first revision of this
module used a PARITY rule -- "a pipe is escaped iff preceded by an odd number of
backslashes" -- inferred from CommonMark's backslash-escape section rather than measured
against a renderer. Under that rule `` `x \\\\| y` `` looks like a live delimiter, and it
reported `G21b` and `R9` as dropping their reachability cells at `553cf2c3`. **They do
not.** cmark-gfm's table-cell scanner is `([\\\\][|] | [^|\\r\\n])*`, so a lone backslash is
consumed as an ordinary character and the last backslash of any run pairs with the pipe;
markdown-it's `escapedSplit()` agrees. Verified on three instruments -- local `cmarkgfm`,
GitHub's `/markdown` API at `mode=gfm`, and the same API at `mode=markdown` -- for runs of
1, 2, 3 and 4 backslashes, with a bare-pipe positive control confirming each instrument
does detect real drops. The base bytes of `G21b` render as **5 cells** with `Reachability
evidence` = *"REACHABLE - every battle, every move above 10 PP ... 24 for `shadowball`, 16
for `earthquake`"* and `Observed` = `no`; `R9` renders as 3 intact cells; the whole file at
`553cf2c3` renders 9 tables with no ragged row. That error reached a pull request and was
caught in review. **A checker for a rendering defect must be validated against a renderer.**

MUTATION BATTERY: 9 applied, 9 caught. Recorded because this repo has found four inert
pins, and "the tests pass" is the same kind of claim this module replaces. Each mutation
was applied to a clean tree, the module run, and the tree restored.

  1. `G37`'s pre-#1166 bytes (20 pipes) reinstated IN PLACE, at the row's own line -> 3
     red (uniformity, the reachability-column check naming the row, the repo-wide
     inventory). This is the historical instance, replayed against the live document.
  2. `G37b`'s pre-#1166 bytes (9 pipes) reinstated in place -> 3 red.
  3. G5's Reachability cell blanked with the row left RECTANGULAR -> 1 red, the C116 §5
     check alone. This is the mutation that proves uniformity is not the whole pin.
  4. The G30 row deleted -> 2 red (inventory, and the per-table row-tuple check).
  5. A new `G52` row added *with* a valid reachability cell -> 2 red. A gap cannot join the
     ledger silently even when it is well formed, which is item 14's actual instruction.
  6. `markdown_corpus()` truncated to one file -> 2 red (membership, inventory).
  7. A new bare `|` added to a real table row of `reports/c121_a5_wake_before_contact.md`
     -> 1 red, the inventory. (First attempt at this mutation landed on a `||`-leading line
     inside a fenced Rust excerpt and correctly did NOT fire, which is what prompted the
     fence handling in `tables()` and its own test below.)
  8. `tables()` made to return nothing -> 10 red. A finder that finds nothing must not be
     a green module.
  9. One quarantined row FIXED in `reports/c19_trace_truant_phase_prediction.md`, lowering
     its count -> 1 red. The inventory is exact in both directions, so an improvement also
     has to be recorded rather than absorbed.

The battery deliberately no longer contains "swap the escape rule for the other one". An
earlier revision did, scoring the GFM-CORRECT rule as a caught mutation, which is how a pin
comes to defend a wrong answer. `TheEscapeRuleMatchesRealRenderersTests` now pins the
correct rule directly, with the pre-#1166 rows as the live-drop witnesses.

SCOPE, and it is deliberately two-tier.

  1. `reports/c138_known_gaps_ledger.md` is held to **exact uniformity in both
     directions**, to an exact row inventory, and to a non-empty rendered Reachability cell
     on every gap row.
  2. Every other markdown document under `docs/`, `reports/` and the repository root is
     held to an **exact inventory of the over-delimited rows it already has**. Six
     documents carry 24 such rows between them at `553cf2c3` -- re-derived under the
     GFM-correct rule, which finds the same 24, because every one of them is a bare pipe
     and none involves a backslash at all. C150 does not fix them (three are outside the
     fidelity program, and one is a 6,000-line permanent ledger), and **none of them falls
     under item 14**: no quarantined table has a `Reachability evidence` column, and
     `docs/engine_divergence_ledger_20260728.md` is a different document from the
     known-gaps ledger. The inventory means a NEW one cannot appear anywhere without
     reddening this module.

     **A KNOWN AND DELIBERATE LIMITATION of tier 2, recorded so the next reader does not
     mistake it for an oversight.** The inventory keys on a PER-FILE COUNT, not on row
     identities or line numbers. That is what makes it robust to unrelated edits -- a PR
     that only shifts line numbers in a quarantined document does not touch it, which is
     why C150 and the concurrent ledger change could both land -- but it also means one row
     FIXED and one NEW offender ADDED **in the same file, in the same change** cancel out
     and this module stays green. Cross-file it is exact in both directions; within a
     single file it is exact only on the total. Tightening it to row identities would trade
     that robustness for brittleness against every line shift, so the count is the
     deliberate choice; the residual hole is one file's simultaneous fix-and-regress.
"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

LEDGER = "reports/c138_known_gaps_ledger.md"

_SKIP_DIRS = frozenset(
    {
        ".git",
        "third_party",
        "node_modules",
        "target",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "dist",
        "build",
    }
)

# A GFM table's delimiter row. Kept deliberately narrow: it is what identifies a block of
# `|`-leading lines as a TABLE, and it supplies the authoritative column count.
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")


# ---------------------------------------------------------------------------
# The checker. Everything below derives from these three functions.
# ---------------------------------------------------------------------------


def unescaped_pipe_positions(line: str) -> list[int]:
    """Indices of every `|` that GFM will treat as a CELL DELIMITER.

    THE RULE IS "ANY PRECEDING BACKSLASH", NOT A PARITY RULE, and that is not an
    inference -- it is what two real renderers do. cmark-gfm's table-cell scanner is
    `([\\\\][|] | [^|\\r\\n])*`: a lone backslash that is not followed by a pipe is consumed
    as an ordinary character, so in a run of backslashes the LAST one always pairs with the
    pipe. markdown-it's `escapedSplit()` agrees.

    Verified rather than reasoned, because an earlier revision of this module got it wrong
    in the opposite direction and the error reached a PR. Runs of 1, 2, 3 and 4 backslashes
    before a pipe inside an inline code span, in a three-column table, render as **three
    cells** on all of: local `cmarkgfm` (`github_flavored_markdown_to_html`), GitHub's
    `/markdown` API with `mode=gfm`, and the same API with `mode=markdown`. The positive
    control -- a bare `|`, no backslash -- drops a cell on every one of them, so the
    instrument does detect real drops.

    So `` `a \\| b` ``, `` `a \\\\| b` `` and `` `a \\\\\\| b` `` are all ONE cell, and only a
    bare `` `a | b` `` is two. A bare pipe is the only way a row in this repo has ever
    gained a delimiter.
    """

    return [
        index
        for index, char in enumerate(line)
        if char == "|" and (index == 0 or line[index - 1] != "\\")
    ]


def row_cells(line: str) -> list[str]:
    """The cells GFM will render for a leading-and-trailing-pipe row, stripped.

    Derived from `unescaped_pipe_positions`, so a row that gained a delimiter yields the
    SHIFTED cells a reader actually sees -- which is the point: this is what proves G21b's
    reachability check left the table rather than merely moved inside it.
    """

    positions = unescaped_pipe_positions(line)
    return [
        line[left + 1 : right].strip()
        for left, right in zip(positions, positions[1:])
    ]


def tables(text: str) -> list[dict]:
    """Every GFM table in `text`, as `{header, separator_line, rows, columns}`.

    A table is a maximal run of lines whose first non-space character is `|` and which
    contains a delimiter row. `columns` is the delimiter row's unescaped-pipe count, i.e.
    the authority every other row in the block is measured against. `rows` is the DATA
    rows only -- the header is carried separately, because the row inventories below are
    inventories of entries and the header is not one; `delimiter_anomalies` still measures
    it, since a header that disagrees with its own separator is the same defect.
    """

    lines = text.split("\n")
    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    # Fenced code is NOT scanned. `reports/c121_a5_wake_before_contact.md` has a Rust
    # excerpt whose continuation lines begin with `||`, and this repo's reports quote
    # protocol lines and `|---|`-shaped output freely; a fence containing a delimiter-row
    # lookalike would otherwise be reported as a ragged table forever.
    fenced = False
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            fenced = not fenced
            if current:
                blocks.append(current)
            current = []
            continue
        if fenced:
            if current:
                blocks.append(current)
            current = []
            continue
        if line.lstrip().startswith("|"):
            current.append((number, line))
        else:
            if current:
                blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    found: list[dict] = []
    for block in blocks:
        separators = [entry for entry in block if _SEPARATOR.match(entry[1])]
        if not separators:
            continue
        separator = separators[0]
        found.append(
            {
                "header": block[0],
                "header_line": block[0][0],
                "separator_line": separator[0],
                "columns": len(unescaped_pipe_positions(separator[1])),
                "rows": [
                    entry
                    for entry in block
                    if entry[0] not in (separator[0], block[0][0])
                ],
            }
        )
    return found


def delimiter_anomalies(text: str) -> dict[str, list[tuple[int, int, int]]]:
    """`(line, measured, expected)` for every row whose delimiter count is off.

    Split into `over` and `under` because they are different defects. `over` is the
    dangerous one: GFM DROPS the surplus cells, so content disappears from the rendered
    document. `under` renders as trailing empty cells, which is visible.
    """

    over: list[tuple[int, int, int]] = []
    under: list[tuple[int, int, int]] = []
    for table in tables(text):
        expected = table["columns"]
        for number, line in [table["header"], *table["rows"]]:
            measured = len(unescaped_pipe_positions(line))
            if measured > expected:
                over.append((number, measured, expected))
            elif measured < expected:
                under.append((number, measured, expected))
    return {"over": over, "under": under}


def markdown_corpus() -> list[str]:
    """Every markdown document under `docs/`, `reports/` and the repository root."""

    found: list[str] = []
    for tree in ("docs", "reports"):
        for root, dirs, files in os.walk(REPO / tree):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in files:
                if name.endswith(".md"):
                    found.append(
                        os.path.relpath(os.path.join(root, name), REPO)
                    )
    for name in sorted(os.listdir(REPO)):
        if name.endswith(".md"):
            found.append(name)
    return sorted(found)


# ---------------------------------------------------------------------------
# The pinned facts about the ledger.
# ---------------------------------------------------------------------------

_EXPECTED_LEDGER_TABLES = 9

# The five §3.x tables that carry a `Reachability evidence` column, keyed by the header
# line's own text so a table being reordered does not silently retarget the pin. Each
# value is the EXACT row inventory, in document order -- not a count, so a gap cannot join
# the ledger without an author touching this module and therefore the column beside it.
_REACHABILITY_HEADER = "| # | Gap | Class | Reachability evidence | Observed |"
_REACHABILITY_COLUMN = 3  # zero-based, within a 5-cell row

_EXPECTED_GAP_ROWS: tuple[tuple[str, ...], ...] = (
    ("**G0**",),
    (
        "**G1**", "**G2**", "**G3**", "**G4**", "**G5**", "**G6**", "**G7**",
        "**G8**", "**G9**", "**G10**", "**G11**", "**G12**", "**G13**", "**G14**",
        "**G15**", "**G16**", "**G17**", "**G18**", "**G19**", "**G20**", "**G21**",
        "**G21b**", "**G22**", "**G23**", "**G24**", "**G25**", "**G26**", "**G27**",
        "**G28**", "**G29**", "**G49**", "**G50**", "**G51**", "**G30**",
    ),
    (
        "**G31**", "**G32**", "**G33**", "**G33b**", "**G34**", "**G35**", "**G36**",
        "**G37**", "**G37b**", "**G38**", "**G39**", "**G40**",
    ),
    (
        "**H1**", "**H2**", "**H3**", "**H4**", "**H5**", "**H5b**", "**H5c**",
        "**H6**", "**H7**", "**H8**", "**H9**", "**H10**", "**H11**", "**H12**",
        "**H13**", "**H14**", "**H15**", "**H16**", "**H17**", "**H18**", "**H19**",
        "**H21**", "**H20**",
    ),
    (
        "**G41**", "**G42**", "**G43**", "**G43b**", "**G43c**", "**G44**", "**G45**",
        "**G46**", "**G47**", "**G48**",
    ),
)

# §4, "Dropped by the reachability filter -- verified UNREACHABLE". Three columns, and the
# third is where the measurement lives, so it is this table's reachability column.
_UNREACHABLE_HEADER = "| # | Candidate | Why it is unreachable, measured |"
_EXPECTED_UNREACHABLE_ROWS: tuple[str, ...] = (
    "**R1**", "**R2**", "**R3**", "**R4**", "**R5**", "**R6**", "**R7**", "**R8**",
    "**R9**", "**R10**", "**R11**", "**R12**", "**R13**", "**R14**", "**R15**",
    "**R16**", "**R17**", "**R18**", "**R19**", "**R20**", "**R21**", "**R22**",
    "**R23**", "**R24**", "**R25**", "~~**R26**~~", "**R27**",
)

# Every gap row's reachability cell is prose, and the shortest of them is `**no**`-adjacent
# housekeeping rather than a measurement, so the pin is non-emptiness plus a floor that the
# shortest real cell clears. Measured at `553cf2c3` after the C150 fix: the minimum is 4
# characters (`H17`'s `n/a.`), so 3 is a floor that no row currently sits on and that a
# placeholder like `-` or `?` would fail.
_MINIMUM_REACHABILITY_CELL = 3

# ---------------------------------------------------------------------------
# The repo-wide quarantine. Exact, per file, so a new offender anywhere is red.
# ---------------------------------------------------------------------------
#
# MEASURED, not reasoned, and re-measured after the delimiter rule was corrected:
# `delimiter_anomalies()` was run over `markdown_corpus()` at `553cf2c3` and again here, under
# both the (wrong) parity rule and the GFM-correct one. Both find the SAME 24 rows in the same
# six files, because every one of them is a BARE pipe -- a protocol literal or a
# `component_mismatch:a|b` counter name written without an escape -- and none involves a
# backslash at all. `reports/c138_known_gaps_ledger.md` appears in neither run.
#
# NONE OF THESE FALLS UNDER ITEM 14, checked rather than assumed: no quarantined table has a
# `Reachability evidence` column, and `docs/engine_divergence_ledger_20260728.md` is a
# different document from the known-gaps ledger (its headers are
# `# | Mechanic | Divergence | Rate | Repro | Status`).
#
# Why these are quarantined rather than fixed here: `docs/engine_divergence_ledger_20260728.md`
# is a 6,000-line permanent ledger, `report.md` and `docs/self_play_convergence_findings.md`
# are outside the fidelity program, and the three `reports/c1*` entries predate C138. Editing
# 24 rows across six documents to satisfy a check introduced in the same commit buries the
# check in an unreviewable diff. The inventory is the control in the meantime: it is EXACT, so
# both a new offender and a fixed one turn this red, and a fix must come with its own
# measurement.
_EXPECTED_OVER_DELIMITED: dict[str, int] = {
    "docs/engine_divergence_ledger_20260728.md": 11,
    "docs/self_play_convergence_findings.md": 2,
    "report.md": 6,
    "reports/c117_validation_holdout_baseline.md": 1,
    "reports/c121_a5_wake_before_contact.md": 1,
    "reports/c19_trace_truant_phase_prediction.md": 3,
}


def _ledger_text() -> str:
    return (REPO / LEDGER).read_text(encoding="utf-8")


def _tables_by_header(text: str) -> dict[str, list[dict]]:
    lines = text.split("\n")
    grouped: dict[str, list[dict]] = {}
    for table in tables(text):
        header = lines[table["header_line"] - 1].rstrip()
        grouped.setdefault(header, []).append(table)
    return grouped


# ---------------------------------------------------------------------------
# 1. The checker is exercised on inputs whose answer is known by construction.
# ---------------------------------------------------------------------------


class TheEscapeRuleMatchesRealRenderersTests(unittest.TestCase):
    def test_any_run_of_backslashes_escapes_the_pipe_that_follows(self) -> None:
        # THE RULE, and the answers are the ones three real renderers give: local
        # `cmarkgfm`, GitHub's /markdown API at mode=gfm, and the same API at
        # mode=markdown. Runs of 1..4 backslashes before a pipe in an inline code span all
        # render as ONE cell, not two. An earlier revision of this module used a PARITY
        # rule instead and reported two rows of the ledger as broken when they render
        # perfectly; that error reached a PR. These four numbers are the fix.
        for backslashes in range(1, 5):
            with self.subTest(backslashes=backslashes):
                row = "| a `x " + "\\" * backslashes + "| y` b |"
                self.assertEqual(
                    len(unescaped_pipe_positions(row)),
                    2,
                    "a pipe preceded by ANY backslash is not a cell delimiter: "
                    "cmark-gfm's cell scanner consumes a lone backslash as an ordinary "
                    "character, so the last backslash of a run pairs with the pipe",
                )

    def test_the_two_rows_a_parity_rule_falsely_condemned_are_intact(self) -> None:
        # Regression pin for the review finding, on the real base bytes of both rows,
        # abridged only in prose away from the defect. Under GFM both are rectangular and
        # G21b's Reachability cell holds its pool check -- which is what the renderers say.
        for table, row, columns in (
            (_FIVE_COLUMN_TABLE, _BASE_G21B, 5),
            (_THREE_COLUMN_TABLE, _BASE_R9, 3),
        ):
            with self.subTest(row=row[:24]):
                self.assertEqual(
                    delimiter_anomalies(table + row + "\n"),
                    {"over": [], "under": []},
                )
                self.assertEqual(len(row_cells(row)), columns)
        cells = row_cells(_BASE_G21B)
        self.assertIn("REACHABLE", cells[_REACHABILITY_COLUMN])
        self.assertEqual(cells[-1], "no")

    def test_a_bare_pipe_inside_inline_code_still_delimits(self) -> None:
        # GFM does NOT protect pipes inside code spans. If this ever passes at 2, every
        # `over` finding below becomes unreliable in the same direction.
        self.assertEqual(
            len(unescaped_pipe_positions("| a `x | y` b |")), 3
        )

    def test_fenced_code_that_looks_like_a_ragged_table_is_not_one(self) -> None:
        # Without this, a report quoting `|---|`-shaped output or a Rust `||` continuation
        # would be pinned as a permanent defect and the quarantine inventory below would be
        # measuring the wrong thing.
        fenced = (
            "```\n"
            "| a | b |\n"
            "|---|---|\n"
            "| one | two | three | four |\n"
            "```\n"
        )
        self.assertEqual(tables(fenced), [])
        self.assertEqual(delimiter_anomalies(fenced), {"over": [], "under": []})
        # ...and the identical text OUTSIDE a fence is still caught.
        self.assertEqual(
            delimiter_anomalies(fenced.replace("```\n", ""))["over"], [(3, 5, 3)]
        )

    def test_the_table_finder_locates_every_table_and_its_separator(self) -> None:
        # A loop over zero tables passes every assertion in this module. This is the
        # anti-vacuity control for the whole ledger class below.
        found = tables(_ledger_text())
        self.assertEqual(len(found), _EXPECTED_LEDGER_TABLES)
        self.assertTrue(all(table["columns"] >= 2 for table in found))
        self.assertTrue(
            all(table["separator_line"] == table["header_line"] + 1 for table in found),
            "a separator that is not directly under its header means the finder grouped "
            "two tables into one block, and every column count below is measured against "
            "the wrong authority",
        )


# ---------------------------------------------------------------------------
# 2. The historical instance, as fixtures. Fire-by-construction.
# ---------------------------------------------------------------------------

# `G37` and `G37b` as they stood at `a587e614^` -- #1166's tree before its own fix --
# abridged in the prose but with every pipe intact. These are BARE pipes in protocol
# literals: the one thing that really does take a delimiter. Rendered through cmarkgfm
# these two rows come back with an EMPTY `Reachability evidence` cell.
_PRE_1166_G37 = (
    "| **G37** | **Attract's empty-immobilization branch is indistinguishable from a "
    "fully-capped boost** - 17 sub-cases. `events.rs` reads the marker and renders "
    "`|cant|<ident>|Attract` or `|cant|<ident>|par` exactly; the older probability-mass "
    "`|cant|..|par|` guess is deleted. Showdown's companion "
    "`|-activate|..|move: Attract|[of] <source>` line stays unrenderable. "
    "| R | **REACHABLE, but only via the ability route.** The move `attract` is 0 of 220; "
    "Cute Charm is on 3 of 220. | no |"
)
_PRE_1166_G37B = (
    "| **G37b** | **NEW, opened by the immobilizer-marker change.** Searched worlds now "
    "emit `|cant|<ident>|Attract`, which `public_action_capture.py` keys as the public "
    "action `cant:attract`. | R | REACHABLE via Cute Charm, as G37. | no |"
)

# The `553cf2c3` bytes of the two rows a parity rule falsely condemned, same abridgement.
# Under the GFM rule both are rectangular; they are the negative controls.
_BASE_G21B = (
    r"| **G21b** | **The engine does not decrement PP above 10 at all.** "
    r"Undercuts **Encore's PP-zero termination** "
    r"(`move_fails_encore(...) \\|\\| move_slot.pp <= 0`). "
    r"| E | REACHABLE - every battle, every move above 10 PP. | no |"
)
_BASE_R9 = (
    r"| **R9** | **Liquid Ooze mislabelled by the residual-heal renderer** | "
    r"rendered as `\\|-damage\\|...\\|[from] ability: Liquid Ooze` before "
    r"`residual_heal_cause` is called. |"
)

_FIVE_COLUMN_TABLE = (
    "| # | Gap | Class | Reachability evidence | Observed |\n|---|---|---|---|---|\n"
)
_THREE_COLUMN_TABLE = (
    "| # | Candidate | Why it is unreachable, measured |\n|---|---|---|\n"
)


class TheHistoricalDropIsCaughtTests(unittest.TestCase):
    def test_the_pre_1166_g37_row_loses_its_reachability_cell(self) -> None:
        anomalies = delimiter_anomalies(_FIVE_COLUMN_TABLE + _PRE_1166_G37 + "\n")
        over = anomalies["over"]
        self.assertEqual(len(over), 1, anomalies)
        line, measured, expected = over[0]
        self.assertEqual((line, expected), (3, 6))
        self.assertGreater(measured, expected)
        # ...and this is WHY it matters. GFM keeps the first five cells and DROPS the rest,
        # so the reachability check is not merely displaced, it is gone.
        rendered = row_cells(_PRE_1166_G37)[: _FIVE_COLUMN_TABLE.count("|") // 2 - 1]
        self.assertNotIn("REACHABLE", rendered[_REACHABILITY_COLUMN])

    def test_the_pre_1166_g37b_row_is_caught_too(self) -> None:
        anomalies = delimiter_anomalies(_FIVE_COLUMN_TABLE + _PRE_1166_G37B + "\n")
        self.assertEqual(len(anomalies["over"]), 1, anomalies)
        self.assertEqual(anomalies["over"][0][0], 3)

    def test_the_rows_1166_shipped_and_the_base_ledger_rows_all_pass(self) -> None:
        # The other half of a witness: it must also be quiet on correct input. `\|` is the
        # fix #1166 applied, and the two base rows use `\\|` -- both are one cell.
        fixed_g37 = _PRE_1166_G37.replace("`|", r"`\|").replace("|`", r"\|`").replace(
            "|<ident>|", r"\|<ident>\|"
        ).replace("|..|", r"\|..\|").replace("|[of]", r"\|[of]")
        for table, row in (
            (_FIVE_COLUMN_TABLE, fixed_g37),
            (_FIVE_COLUMN_TABLE, _BASE_G21B),
            (_THREE_COLUMN_TABLE, _BASE_R9),
        ):
            with self.subTest(row=row[:24]):
                self.assertEqual(
                    delimiter_anomalies(table + row + "\n"),
                    {"over": [], "under": []},
                )

    def test_an_empty_reachability_cell_is_flagged(self) -> None:
        # Uniformity alone does not satisfy C116 §5. A row can be perfectly rectangular
        # and still record no pool check, so the ledger pin below checks the cell too --
        # and this is the input that proves that check is not vacuous.
        row = "| **GX** | a new gap | E |  | no |"
        self.assertEqual(delimiter_anomalies(_FIVE_COLUMN_TABLE + row + "\n")["over"], [])
        self.assertEqual(row_cells(row)[_REACHABILITY_COLUMN], "")

    def test_a_row_with_too_few_cells_is_flagged(self) -> None:
        row = "| **GX** | a new gap | E | REACHABLE |"
        anomalies = delimiter_anomalies(_FIVE_COLUMN_TABLE + row + "\n")
        self.assertEqual(anomalies["under"], [(3, 5, 6)], anomalies)


# ---------------------------------------------------------------------------
# 3. The ledger itself.
# ---------------------------------------------------------------------------


class TheKnownGapsLedgerTests(unittest.TestCase):
    def test_every_table_in_the_ledger_is_exactly_uniform(self) -> None:
        anomalies = delimiter_anomalies(_ledger_text())
        self.assertEqual(
            anomalies,
            {"over": [], "under": []},
            f"{LEDGER} has a row whose delimiter count differs from its header's. An "
            "`over` row has cells DROPPED by GFM -- if it is a gap row, its "
            "pool-reachability check may have left the rendered table, which is C116 §5's "
            "standing rule broken invisibly. The cause is a BARE `|` in the row's prose; "
            "escape it as `\\|`.",
        )

    def test_the_ledger_writes_escaped_pipes_with_exactly_one_backslash(self) -> None:
        # TYPOGRAPHY, NOT GEOMETRY, and the distinction is the whole review finding above.
        # `\\|` does NOT take a delimiter -- cmark-gfm and GitHub's API both render the row
        # rectangular -- so this pin guards nothing about item 14. What `\\|` does do is
        # render a STRAY BACKSLASH inside the code span: `` `a \\| b` `` comes back as
        # `a \| b`, where the author meant `a | b`. Verified through cmarkgfm, not assumed.
        # Kept because it is a real (if cosmetic) defect and free to check; labelled so the
        # next reader does not mistake it for a rendering-integrity control.
        #
        # Scoped to TABLE ROWS rather than the file, and the two measurements differ:
        # whole document at head, 43 single-backslash and **1** double-backslash (the §8
        # prose that has to be able to write the sequence it describes); inside table rows,
        # 43 and **0**. An earlier revision quoted the row-scoped pair as a whole-file
        # measurement -- an unscoped negative inside the sentence claiming it was scoped,
        # which is this program's signature defect in miniature.
        text = _ledger_text()
        rows = "\n".join(
            line
            for table in tables(text)
            for _, line in [table["header"], *table["rows"]]
        )
        doubled = len(re.findall(r"(?<!\\)\\\\\|", rows))
        single = len(re.findall(r"(?<!\\)\\\|", rows))
        self.assertEqual(
            doubled,
            0,
            "a table row writes an escaped pipe with TWO backslashes. It still renders as "
            "one cell, but the code span comes out with a literal backslash in it.",
        )
        # Anti-vacuity: the assertion above is only meaningful in a document that actually
        # uses the convention. If this ever reaches 0 the document stopped escaping pipes
        # and the check above has nothing left to distinguish.
        self.assertGreater(single, 20, "the ledger stopped using the `\\|` convention")

    def test_every_gap_row_records_a_pool_reachability_check_in_its_own_column(self) -> None:
        grouped = _tables_by_header(_ledger_text())
        found = grouped.get(_REACHABILITY_HEADER, [])
        self.assertEqual(
            len(found), len(_EXPECTED_GAP_ROWS),
            "the number of `Reachability evidence` tables changed; the inventory below "
            "and this check are both keyed on it",
        )
        for table, expected_rows in zip(found, _EXPECTED_GAP_ROWS):
            for number, line in table["rows"]:
                cells = row_cells(line)
                with self.subTest(line=number, row=cells[0] if cells else "?"):
                    # Read the cell AT THE RENDERED POSITION, not by counting from the
                    # right. That is the whole difference between this pin and a grep.
                    self.assertEqual(len(cells), 5)
                    check = cells[_REACHABILITY_COLUMN]
                    self.assertGreaterEqual(
                        len(check), _MINIMUM_REACHABILITY_CELL,
                        f"{LEDGER}:{number} ({cells[0]}) has no pool-reachability check "
                        "in its Reachability evidence column. C116 item 14: no entry "
                        "joins the ledger without one recorded next to it.",
                    )
                    self.assertTrue(cells[-1], f"{LEDGER}:{number} has a blank Observed cell")
            self.assertEqual(
                tuple(row_cells(line)[0] for _, line in table["rows"]),
                expected_rows,
            )

    def test_the_gap_row_inventory_is_exactly_pinned(self) -> None:
        # Exact, and in document order. A gap added, removed or renumbered fails here,
        # which is the point: it forces the author through the column beside it.
        grouped = _tables_by_header(_ledger_text())
        inventory = tuple(
            tuple(row_cells(line)[0] for _, line in table["rows"])
            for table in grouped.get(_REACHABILITY_HEADER, [])
        )
        self.assertEqual(inventory, _EXPECTED_GAP_ROWS)
        self.assertEqual(sum(len(rows) for rows in inventory), 80)

    def test_every_unreachable_candidate_row_records_its_measurement(self) -> None:
        grouped = _tables_by_header(_ledger_text())
        found = grouped.get(_UNREACHABLE_HEADER, [])
        self.assertEqual(len(found), 1)
        table = found[0]
        self.assertEqual(
            tuple(row_cells(line)[0] for _, line in table["rows"]),
            _EXPECTED_UNREACHABLE_ROWS,
        )
        for number, line in table["rows"]:
            cells = row_cells(line)
            with self.subTest(line=number, row=cells[0]):
                self.assertEqual(len(cells), 3)
                self.assertGreaterEqual(
                    len(cells[2]), _MINIMUM_REACHABILITY_CELL,
                    f"{LEDGER}:{number} ({cells[0]}) is filed UNREACHABLE with no "
                    "measurement recorded next to it",
                )


# ---------------------------------------------------------------------------
# 4. The repo-wide inventory, so a new offender cannot appear elsewhere either.
# ---------------------------------------------------------------------------


class RepoWideTableIntegrityInventoryTests(unittest.TestCase):
    def test_the_corpus_contains_the_ledger_and_every_quarantined_document(self) -> None:
        # The fail-open this pin has: a document dropping out of the walk makes the
        # inventory below EASIER to satisfy. Naming the seven that matter closes it
        # without pinning a whole-corpus file count that any unrelated `.md` would move.
        corpus = markdown_corpus()
        self.assertIn(LEDGER, corpus)
        for name in _EXPECTED_OVER_DELIMITED:
            self.assertIn(name, corpus)
        self.assertGreater(len(corpus), 100, f"the markdown walk collapsed: {corpus}")

    def test_the_inventory_of_over_delimited_rows_is_exactly_pinned(self) -> None:
        measured: dict[str, int] = {}
        for name in markdown_corpus():
            text = (REPO / name).read_text(encoding="utf-8")
            count = len(delimiter_anomalies(text)["over"])
            if count:
                measured[name] = count
        self.assertEqual(
            measured,
            _EXPECTED_OVER_DELIMITED,
            "the inventory of markdown rows whose surplus cells GFM DROPS has changed. A "
            "new entry is a new rendering defect -- escape the pipe with `\\|`. A "
            "vanished or smaller entry means one was fixed: good, but update the "
            f"inventory in the same change. {LEDGER} must never appear here.",
        )
        self.assertNotIn(LEDGER, measured)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
