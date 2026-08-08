"""Re-derive the known-gaps ledger's table geometry, escape-aware, instead of asserting it.

WHY THIS MODULE EXISTS. C116 Phase 4 item 14's standing instruction is that *"no entry
joins the known-gaps ledger without a pool-reachability check recorded next to it"*
(`reports/c124_a6_is_knowable.md` §2). `reports/c138_known_gaps_ledger.md` records that
check in a dedicated **Reachability evidence** column -- and a table row with more cell
delimiters than its header does not merely look wrong in GFM, it **silently drops the
overflow**. So an unescaped `|` anywhere in a row's prose can delete that row's
reachability check from the rendered document while leaving it in the bytes, which is the
worst available failure: the ledger passes inspection by grep and fails item 14 in the
only form a reader sees.

That is not hypothetical. It happened twice and survived two audits:

  * `G21b` shipped in #1151 with `` `move_fails_encore(...) \\|\\| move_slot.pp <= 0` `` --
    a *double* backslash, which escapes the backslash and leaves the pipe live. The row
    carried **8** delimiters against a 6-column header, so GFM rendered `\\` as the Class
    cell, the tail of the description as **Reachability evidence**, `E` as **Observed**,
    and threw the row's actual pool check (*"REACHABLE -- every battle, every move above
    10 PP ... Gen3 max PP ... 24 for shadowball, 16 for earthquake"*) and its `no` value
    away entirely.
  * `R9` shipped in the same commit with `` `\\|-damage\\|...\\|[from] ability: Liquid Ooze` ``
    -- **7** delimiters against a 4-column header.

  * And `reports/c146_negative_claim_audit.md` then asserted *"all 9 tables now have
    uniform column counts, verified by re-deriving delimiter counts per row"* while BOTH
    rows were non-uniform, at the very commit that says it. A re-derivation that treats
    `\\|` as an escaped pipe sees a uniformity that is not there. That is the whole reason
    this check is code and not a sentence: the previous fix's own verification claim was
    the thing that failed, and the disposition recorded next to it was *"Not pinned ...
    filed as a follow-up instead."* A filed follow-up is not a control.

MUTATION BATTERY: 10 applied, 10 caught. Recorded because this repo has found four inert
pins, and "the tests pass" is the same kind of claim as the one this module exists to
replace. Each mutation was applied to a clean tree, the module run, and the tree restored.

  1. G21b reverted to its pre-fix `\\|\\|` -> 4 red (uniformity, the `\\|` convention, the
     reachability-column check naming line 242, and the repo-wide inventory).
  2. R9 reverted to its pre-fix `\\|` -> 4 red, including the §4 measurement check at 465.
  3. G5's Reachability cell blanked with the row left RECTANGULAR -> 1 red, the C116 §5
     check alone. This is the mutation that proves uniformity is not the whole pin.
  4. The G30 row deleted -> 2 red (inventory, and the per-table row-tuple check).
  5. A new `G52` row added *with* a valid reachability cell -> 2 red. A gap cannot join the
     ledger silently even when it is well formed, which is item 14's actual instruction.
  6. `unescaped_pipe_positions` "simplified" to `line[index - 1] != "\\"` -- i.e. any
     preceding backslash escapes -- -> 3 red, both pre-fix row witnesses among them. This
     is the exact mis-instrument that made C146's uniformity claim look true.
  7. `markdown_corpus()` truncated to one file -> 2 red (membership, inventory).
  8. A new unescaped `|` added to a real table row of `reports/c121_a5_wake_before_contact.md`
     -> 1 red, the inventory. (First attempt at this mutation landed on a `||`-leading line
     inside a fenced Rust excerpt and correctly did NOT fire, which is what prompted the
     fence handling in `tables()` and its own test below.)
  9. `tables()` made to return nothing -> 8 red. A finder that finds nothing must not be a
     green module.
 10. One quarantined row FIXED in `reports/c19_trace_truant_phase_prediction.md`, lowering
     its count -> 1 red. The inventory is exact in both directions, so an improvement also
     has to be recorded rather than absorbed.

THIS REPO HAS FOUND FOUR INERT PINS -- tests that exist and guard nothing -- so the
witness class below is not decoration. `TheTwoRowsThisModuleWasWrittenForTests` feeds the
checker the **pre-fix bytes of both rows, verbatim**, and asserts it reports exactly 8 and
exactly 7 delimiters; feeds it the fixed bytes and asserts the Reachability cell comes back
with the pool check in it; and feeds it a blank reachability cell and a short row and
asserts both are flagged. If the escape matcher is ever "simplified" into one that treats
`\\|` as escaped, those pins go red before the ledger pins do.

SCOPE, and it is deliberately two-tier.

  1. `reports/c138_known_gaps_ledger.md` is held to **exact uniformity in both
     directions**, to the single-backslash convention, to an exact row inventory, and to a
     non-empty rendered Reachability cell on every gap row.
  2. Every other markdown document under `docs/`, `reports/` and the repository root is
     held to an **exact inventory of the over-delimited rows it already has**. Six
     documents carry 24 such rows between them at `553cf2c3`; C150 does not fix them (they
     are three unrelated documents plus a 6,000-line permanent ledger, and rewriting rows
     in those belongs in their own reviewable change), but the inventory means a NEW one
     cannot appear anywhere without reddening this module. A floor would not do that, and
     neither would scoping the check to one file -- which is how C146's claim came to be
     stated repo-wide from a one-file measurement.
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

    A pipe is escaped iff it is preceded by an ODD number of consecutive backslashes.
    `\\|` escapes it; `\\\\|` escapes the backslash and leaves the pipe live; `\\\\\\|`
    escapes it again. GFM honours the backslash escape inside code spans too, which is why
    `` `a \\| b` `` is one cell and `` `a | b` `` is two -- the single most common way a
    row in this repo has gained a delimiter.
    """

    positions: list[int] = []
    for index, char in enumerate(line):
        if char != "|":
            continue
        backslashes = 0
        scan = index - 1
        while scan >= 0 and line[scan] == "\\":
            backslashes += 1
            scan -= 1
        if backslashes % 2 == 0:
            positions.append(index)
    return positions


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
# MEASURED, not reasoned: `delimiter_anomalies()` was run over `markdown_corpus()` on a
# worktree of `origin/main` at `553cf2c3` and again here. The base tree reports the ledger
# at `{"over": 2}` (G21b and R9) and these six files unchanged; this tree reports the ledger
# absent and the six identical. 24 over-delimited rows across six documents, none of them
# in the fidelity program's ledger.
#
# Why these are quarantined rather than fixed here: `docs/engine_divergence_ledger_20260728.md`
# is a 6,000-line permanent ledger, `report.md` and `docs/self_play_convergence_findings.md`
# are outside the fidelity program, and the three `reports/c1*` entries predate C138. Editing
# rows in a permanent ledger to satisfy a check introduced in the same commit is how a
# verification claim gets written before it is true, which is the defect C150 is closing.
# The inventory is the control in the meantime: it is EXACT, so both a new offender and a
# fixed one turn this red, and a fix must come with its own measurement.
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


class TheEscapeAwareDelimiterCounterIsItselfExercisedTests(unittest.TestCase):
    def test_a_single_backslash_escapes_a_pipe(self) -> None:
        # The document's convention. Two delimiters, not three.
        self.assertEqual(
            len(unescaped_pipe_positions(r"| a \| b |")), 2
        )

    def test_a_double_backslash_does_not_escape_the_pipe_that_follows(self) -> None:
        # THE BUG. `\\` escapes the backslash; the pipe stays live. Three, not two.
        self.assertEqual(
            len(unescaped_pipe_positions(r"| a \\| b |")), 3
        )

    def test_a_triple_backslash_escapes_again(self) -> None:
        # Odd number of backslashes -> escaped. A parity rule, not a "starts with \\" rule.
        self.assertEqual(
            len(unescaped_pipe_positions(r"| a \\\| b |")), 2
        )

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
# 2. The two rows this module was written for, as fixtures. Fire-by-construction.
# ---------------------------------------------------------------------------

# The pre-fix bytes, verbatim from `f876803e`, abridged only in the prose either side of
# the defect -- the defect itself and every delimiter is intact.
_PREFIX_G21B = (
    r"| **G21b** | **The engine does not decrement PP above 10 at all.** "
    r"Undercuts **Encore's PP-zero termination** "
    r"(`move_fails_encore(...) \\|\\| move_slot.pp <= 0`). "
    r"| E | REACHABLE - every battle, every move above 10 PP. | no |"
)
_FIXED_G21B = _PREFIX_G21B.replace(r"\\|", r"\|")

_PREFIX_R9 = (
    r"| **R9** | **Liquid Ooze mislabelled by the residual-heal renderer** | "
    r"rendered as `\\|-damage\\|...\\|[from] ability: Liquid Ooze` before "
    r"`residual_heal_cause` is called. |"
)
_FIXED_R9 = _PREFIX_R9.replace(r"\\|", r"\|")

_SIX_COLUMN_TABLE = "| # | Gap | Class | Reachability evidence | Observed |\n|---|---|---|---|---|\n"
_FOUR_COLUMN_TABLE = "| # | Candidate | Why it is unreachable, measured |\n|---|---|---|\n"


class TheTwoRowsThisModuleWasWrittenForTests(unittest.TestCase):
    def test_the_prefix_g21b_row_is_flagged_with_eight_delimiters(self) -> None:
        anomalies = delimiter_anomalies(_SIX_COLUMN_TABLE + _PREFIX_G21B + "\n")
        self.assertEqual(anomalies["over"], [(3, 8, 6)], anomalies)
        # ...and this is WHY it matters: the rendered Reachability cell is not the check.
        cells = row_cells(_PREFIX_G21B)
        self.assertEqual(len(cells), 7)
        self.assertNotIn("REACHABLE", cells[_REACHABILITY_COLUMN])
        self.assertEqual(cells[4], "E", "the Observed column renders the Class value")

    def test_the_prefix_r9_row_is_flagged_with_seven_delimiters(self) -> None:
        anomalies = delimiter_anomalies(_FOUR_COLUMN_TABLE + _PREFIX_R9 + "\n")
        self.assertEqual(anomalies["over"], [(3, 7, 4)], anomalies)

    def test_the_fixed_rows_pass_and_render_their_reachability_cell(self) -> None:
        for table, row, columns in (
            (_SIX_COLUMN_TABLE, _FIXED_G21B, 6),
            (_FOUR_COLUMN_TABLE, _FIXED_R9, 4),
        ):
            with self.subTest(row=row[:24]):
                anomalies = delimiter_anomalies(table + row + "\n")
                self.assertEqual(anomalies, {"over": [], "under": []})
                cells = row_cells(row)
                self.assertEqual(len(cells), columns - 1)
        cells = row_cells(_FIXED_G21B)
        self.assertIn("REACHABLE", cells[_REACHABILITY_COLUMN])
        self.assertEqual(cells[-1], "no")

    def test_an_empty_reachability_cell_is_flagged(self) -> None:
        # Uniformity alone does not satisfy C116 §5. A row can be perfectly rectangular
        # and still record no pool check, so the ledger pin below checks the cell too --
        # and this is the input that proves that check is not vacuous.
        row = "| **GX** | a new gap | E |  | no |"
        self.assertEqual(delimiter_anomalies(_SIX_COLUMN_TABLE + row + "\n")["over"], [])
        self.assertEqual(row_cells(row)[_REACHABILITY_COLUMN], "")

    def test_a_row_with_too_few_cells_is_flagged(self) -> None:
        row = "| **GX** | a new gap | E | REACHABLE |"
        anomalies = delimiter_anomalies(_SIX_COLUMN_TABLE + row + "\n")
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
            f"{LEDGER} has a row whose escape-aware delimiter count differs from its "
            "header's. An `over` row has cells DROPPED by GFM -- if it is a gap row, its "
            "pool-reachability check may have left the rendered table, which is C116 §5's "
            "standing rule broken invisibly. The usual cause is `\\\\|` where the "
            "convention is `\\|`.",
        )

    def test_the_ledger_uses_only_the_single_backslash_pipe_convention(self) -> None:
        # Scoped to TABLE ROWS, not the whole document, and deliberately: §8 now records
        # this defect class by name and has to be able to write the offending sequence in
        # prose. Outside a table row a `\\|` renders a backslash and a pipe and breaks
        # nothing, so a document-wide ban would be a rule wider than the harm -- the same
        # over-broad-claim shape this module exists to catch.
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
            "`\\\\|` renders a literal backslash and leaves the pipe live. This is the "
            "exact byte sequence that broke G21b and R9 in #1151 and that C146's "
            "uniformity claim did not see.",
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
