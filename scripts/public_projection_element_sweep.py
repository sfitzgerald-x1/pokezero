"""Per-element mutation sweep over ``src/pokezero/public_projection.py``.

WHY THIS FILE EXISTS AT ALL, and it is the whole point: this sweep has been the
headline deliverable of three merged PRs (#1223, #1225, #1226) and its number was
published five different ways while the generator lived in ``/tmp`` on one machine.
Three agents produced three denominators for "the same" sweep; they reconcile only
because the runs happened to share a box. This program's own rule -- *a digest is
published with the script that produces it, never with a description of it* -- was
earned on a records hash and never applied to counts. **A survivor count is exactly
as unreproducible as a hash when its generator is uncommitted.**

So: the universe, the element threshold and the PROVENANCE of every recorded pair
are data in this file, and every count is printed next to

  * the universe name and its exact definition,
  * the element threshold,
  * the sha256 of the file that was mutated,
  * a fingerprint of the kill criterion's CONTENT, not just its command line, and
  * the publication a pair came from, or the word DERIVED if it came from nobody.

A number with no publication behind it is labelled DERIVED and is never reported as
matching anything. The first revision of this script got that wrong: it carried
91/13, 107/1 and 119/13 as "recorded" pairs. None of the three was ever published as
a (deletions, survivors) pair -- see TRAP 1 -- so the script printed MATCHES RECORDED
over a reconciliation of digits whose published meanings differed. That is this
file's own TRAP 1 recurring one level up, in the file written to prevent it, and it
is why provenance is now typed rather than described.

The record, re-derived from the merged PR bodies and from git
------------------------------------------------------------

    commit    public_projection.py  tests/test_public_projection.py  test defs
    fb600899  87867c2e              f22ea661                         100   <- #1223 merge = #1225's base arm
    6af47d25  8bbebad8              fe118a30                         113   <- #1225 merge = #1226's base arm
    6612054a  8bbebad8              65a4e53b                         114   <- #1226 merge
    b0d21647  8bbebad8              65a4e53b                         114   <- main at time of writing

    for c in fb600899 6af47d25 6612054a b0d21647; do
      echo "$c $(git show "${c}:src/pokezero/public_projection.py" | shasum -a 256 | cut -c1-8)" \
           "$(git show "${c}:tests/test_public_projection.py" | shasum -a 256 | cut -c1-8)"; done

Every pair below is anchored to one of those target digests. The pre-#1225 file
(``87867c2e``) is a DIFFERENT FILE from this one, so pairs measured on it are
reported as NOT COMPARABLE rather than as a mismatch.

TRAP 1 -- the collisions, which is why a bare number identifies nothing
----------------------------------------------------------------------

* **91** is the Class A DENOMINATOR on the pre-#1225 file, published with **25**
  survivors (#1225's base arm). It is also what this script measures for Class A +
  ``__all__`` on the post file. Same digits, different universe, different file, and
  a survivor count that differs by 12.
* **107** is a SURVIVOR COUNT -- of the 225-deletion every-string-collection universe
  on the pre file (#1225). It is also 79 + 28 module-level dict keys on the post
  file. A denominator and a survivor count are not the same kind of thing.
* **119** is a Class A denominator published for the pre file by the lineage before
  #1225, which **#1225 could not reproduce** -- its enumerator scored that class at
  91 with an identical survivor set, so the disagreement was in what else was
  enumerated. It is recorded here as DISPUTED, not as a result.

TRAP 2 -- the 93-vs-91 gap is an element threshold of >=1 instead of >=2
-----------------------------------------------------------------------

It admits exactly two more mutants, ``("struggle",)`` and
``["attribution_unsafe"]``, both trivial survivors, so a threshold slip inflates
both counts by 2 and reads like a coverage regression. ``--list`` prints them.

TRAP 3 -- the kill criterion moves too
--------------------------------------

Survivors are a function of the test suite, and ``tests/test_public_projection.py``
grew 100 -> 113 -> 114 across these three PRs. #1225's post-file pairs were measured
at criterion ``fe118a30`` (113 test defs); this tree is ``65a4e53b`` (114). #1226
measured 93/15 at BOTH of those criteria and got the same answer, which is the only
reason a post-#1225 pair can be checked here at all. The criterion's content
fingerprint is printed and carried on every count line.

What a mutant is
----------------

One mutant deletes ONE element from ONE collection literal, or one ``key: value``
pair from one dict literal, and runs the kill criterion. Deletion is surgical text
removal that PRESERVES THE LINE COUNT (removed spans are replaced by their own
newlines), because this file's neighbours are cited by line elsewhere in the tree
and a mutant that dies of a citation gate has died of the wrong cause.

Hazards from report 4 sections 4.2 and 4.4, handled in code rather than in a README:

* **A mutant that was never applied scores as a kill.** Every mutant is re-parsed and
  diffed against the original literal inventory before the tests run. A mutant that
  does not remove exactly the intended element -- a 2-tuple degenerating into a plain
  string, or a deletion that leaves the file unparseable -- is reported ``invalid``,
  never as a kill.
* **Stale bytecode.** ``__pycache__`` under ``src/`` and ``tests/`` is purged once at
  start and every child runs ``python -B``.
* **A mutant dying of the wrong cause.** Deaths are classified from the child's output
  and reported separately from kills by assertion.
* **An instrument that cannot report failure reports success.** The worktree check
  FAILS CLOSED -- if git cannot be run, the sweep refuses rather than measuring a
  modified file and printing a match -- and it covers the kill criterion's own files
  AND the arm-selection files (``ARM_SELECTION_FILES``), not just the target. The
  target digest is CHECKED against the digests published pairs were measured on, not
  merely displayed, and ``git status --porcelain`` over the WHOLE worktree is printed
  verbatim so an edit outside the watched set is at least legible in the artifact.

Usage
-----

    python scripts/public_projection_element_sweep.py --list
    python scripts/public_projection_element_sweep.py                # every universe
    python scripts/public_projection_element_sweep.py -u class_a
    python scripts/public_projection_element_sweep.py --structural-only

The sweep mutates the working tree in place and restores it byte-for-byte, printing
the sha256 before and after.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import string
import subprocess
import sys
import time
import tokenize
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The file under mutation. Every figure this script prints is about THIS file at the
#: sha256 printed with it, and about nothing else.
TARGET = Path("src/pokezero/public_projection.py")

#: The kill criterion. It is a command AND a set of files whose content is
#: fingerprinted; `tests.test_public_projection` is the module that owns the target.
#: A wider selection is a different criterion and its digest will say so.
DEFAULT_TEST_TARGETS = ("tests.test_public_projection",)

#: Every target digest a published pair below was measured on. CHECKED, not displayed.
KNOWN_TARGET_DIGESTS = {
    "87867c2e": "the PRE-#1225 file (fb600899, #1225's base arm)",
    "8bbebad8": "the POST-#1225 file (6af47d25 through b0d21647)",
}

#: The files that decide WHICH TREE is measured, and therefore part of the
#: measurement. ``tests/__init__.py`` and ``tests/conftest.py`` both move THIS
#: checkout's ``src`` to the front of ``sys.path``; report 4 section 4.2 item 3 is a
#: whole review round voided by exactly that mechanism. Watching only the target and
#: the criterion module left them unwatched, so the readout could print
#: `worktree: verified clean` and `same criterion` on a tree whose arm selection had
#: been edited -- section 4.2 item 3 one level up, inside the provenance instrument.
#: Wrong-tree poisoning would still surface downstream (every mutant survives), but a
#: coverage-WEAKENING edit here need not, so these are watched and fingerprinted.
ARM_SELECTION_FILES = (Path("tests/__init__.py"), Path("tests/conftest.py"))


@dataclass(frozen=True)
class Publication:
    """A (deletions, survivors) pair AS PUBLISHED, with what it was measured on.

    ``status`` is ``published`` or ``disputed``. A disputed pair is one a later run
    could not reproduce; it is kept because deleting it is how a retracted number
    comes back as folklore.
    """

    pr: int
    label: str
    arm: str
    target_sha256: str
    criterion_sha256: str
    criterion_test_defs: int
    deletions: int
    survivors: int
    status: str = "published"
    note: str = ""


@dataclass(frozen=True)
class Universe:
    """A named question. Counts are only meaningful attached to one of these."""

    name: str
    #: Minimum elements a collection literal must have to be admitted. >=2 is Class A;
    #: >=1 admits the two single-element literals and is TRAP 2.
    min_elements: int
    #: Whether the ``__all__`` export list is admitted.
    include_dunder_all: bool
    #: ``none`` | ``module`` (module-level assignments only) | ``all`` (any nesting).
    dict_key_scope: str
    definition: str
    publications: tuple[Publication, ...] = ()
    note: str = ""

    @property
    def derived(self) -> bool:
        return not any(p.status == "published" for p in self.publications)


UNIVERSES: tuple[Universe, ...] = (
    Universe(
        name="class_a",
        min_elements=2,
        include_dunder_all=False,
        dict_key_scope="none",
        definition=">=2-element string-literal collections, __all__ EXCLUDED, no dict keys",
        publications=(
            Publication(
                pr=1225, label="the reviewer's class", arm="fb600899 (base arm)",
                target_sha256="87867c2e", criterion_sha256="f22ea661", criterion_test_defs=100,
                deletions=91, survivors=25,
                note="the PRE file. 91 here is a DENOMINATOR published with 25 survivors",
            ),
            Publication(
                pr=1225, label="the reviewer's class", arm="6af47d25 (PR arm)",
                target_sha256="8bbebad8", criterion_sha256="fe118a30", criterion_test_defs=113,
                deletions=79, survivors=1,
                note="the headline pair this script must reproduce on the current tree",
            ),
            Publication(
                pr=0, label="the pre-#1225 lineage's Class A denominator", arm="fb600899 (base arm)",
                target_sha256="87867c2e", criterion_sha256="f22ea661", criterion_test_defs=100,
                deletions=119, survivors=25, status="disputed",
                note="#1225 could not reproduce 119; its enumerator scored the same class at 91 "
                     "with an IDENTICAL survivor set, so only the denominator was in dispute",
            ),
        ),
    ),
    Universe(
        name="ge1_plus_all",
        min_elements=1,
        include_dunder_all=True,
        dict_key_scope="none",
        definition=">=1-element string-literal collections, __all__ INCLUDED, no dict keys",
        publications=(
            Publication(
                pr=1226, label="#1226's own enumeration (25 collections)", arm="6af47d25 (base arm)",
                target_sha256="8bbebad8", criterion_sha256="fe118a30", criterion_test_defs=113,
                deletions=93, survivors=15,
                note="#1226 measured the identical survivor list on its change arm too "
                     "(criterion 65a4e53b, 114 defs), which is why a post-#1225 pair is "
                     "checkable on this tree at all",
            ),
        ),
        note="TRAP 2: 93 - 91 is a THRESHOLD slip, admitting ('struggle',) and ['attribution_unsafe']",
    ),
    Universe(
        name="every_string_collection",
        min_elements=1,
        include_dunder_all=True,
        dict_key_scope="all",
        definition=(">=1-element string-literal collections + __all__ + EVERY dict string key "
                    "at any nesting"),
        publications=(
            Publication(
                pr=1225, label="every string collection, including dict keys and __all__",
                arm="fb600899 (base arm)",
                target_sha256="87867c2e", criterion_sha256="f22ea661", criterion_test_defs=100,
                deletions=225, survivors=107,
                note="the PRE file. 107 here is a SURVIVOR COUNT, not a denominator",
            ),
            Publication(
                pr=1225, label="every string collection, including dict keys and __all__",
                arm="6af47d25 (PR arm)",
                target_sha256="8bbebad8", criterion_sha256="fe118a30", criterion_test_defs=113,
                deletions=213, survivors=83,
                note="the widest genuinely published pair, and the reason dict keys are scoped "
                     "`all` rather than `module`: 81 + 12 + 120 = 213",
            ),
        ),
    ),
    # --- DERIVED: measured here, published by nobody. Each is kept because its
    # --- denominator collides with a published number of a different meaning, and
    # --- printing the derivation beside the collision is the point of this file.
    Universe(
        name="class_a_plus_all",
        min_elements=2,
        include_dunder_all=True,
        dict_key_scope="none",
        definition=">=2-element collections + __all__, no dict keys",
        note="DERIVED. Its denominator is 91, which collides with #1225's PRE-file Class A "
             "denominator -- published with 25 survivors, on a different file",
    ),
    Universe(
        name="class_a_plus_module_dict_keys",
        min_elements=2,
        include_dunder_all=False,
        dict_key_scope="module",
        definition=">=2-element collections + MODULE-LEVEL dict string keys, no __all__",
        note="DERIVED. Its denominator is 79 + 28 = 107, which collides with #1225's PRE-file "
             "SURVIVOR count of 107. A survivor count is not a denominator",
    ),
    Universe(
        name="class_a_plus_all_plus_module_dict_keys",
        min_elements=2,
        include_dunder_all=True,
        dict_key_scope="module",
        definition=">=2-element collections + __all__ + MODULE-LEVEL dict string keys",
        note="DERIVED. Its denominator is 91 + 28 = 119, which collides with the DISPUTED "
             "pre-#1225 Class A denominator that #1225 failed to reproduce",
    ),
)

UNIVERSES_BY_NAME = {universe.name: universe for universe in UNIVERSES}


def _validate_publications() -> None:
    """Every publication must be anchored to a KNOWN target digest, at import time.

    Comparability is decided by ``digest.startswith(publication.target_sha256)``, and
    ``"".startswith`` is vacuously true -- an empty or typo'd prefix would make a pair
    comparable on EVERY file, which is the same class of defect as the invented pairs
    this provenance layer exists to stop. So the anchor is checked here rather than
    trusted: >= 8 lower-case hex characters, and a key of ``KNOWN_TARGET_DIGESTS``.
    """

    def refuse(message: str) -> None:
        # Exit 2, like every other refusal: 0 = measured and matching, 1 = MISMATCH,
        # 2 = refused or misconfigured. A bare AssertionError would surface as 1 and
        # read to a caller as "the sweep disagreed with the record".
        print(f"ERROR: publication table is invalid -- {message}", file=sys.stderr)
        raise SystemExit(2)

    for universe in UNIVERSES:
        for publication in universe.publications:
            anchor = publication.target_sha256
            where = f"{universe.name} / #{publication.pr or '?'} ({publication.arm})"
            if len(anchor) < 8 or anchor.strip(string.hexdigits) or anchor != anchor.lower():
                refuse(f"{where}: target_sha256 {anchor!r} is not >=8 lower-case hex characters")
            if anchor[:8] not in KNOWN_TARGET_DIGESTS:
                refuse(f"{where}: target_sha256 {anchor!r} names a file absent from "
                       f"KNOWN_TARGET_DIGESTS, so nothing states which file it was measured on")
            if publication.status not in ("published", "disputed"):
                refuse(f"{where}: unknown status {publication.status!r}")


_validate_publications()


# --- source surgery -----------------------------------------------------------


def _char_offsets(source: str) -> list[int]:
    """Start offset of every 1-based line, in CHARACTERS."""

    offsets = [0, 0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _offset(lines: list[str], starts: list[int], lineno: int, col: int) -> int:
    """Absolute character offset of an ``ast`` (lineno, col_offset) pair.

    ``col_offset`` is a UTF-8 BYTE offset and this file contains non-ASCII prose, so
    the prefix is decoded rather than sliced.
    """

    prefix = lines[lineno - 1].encode("utf-8")[:col].decode("utf-8")
    return starts[lineno] + len(prefix)


@dataclass(frozen=True)
class Candidate:
    """One deletable element."""

    kind: str  # "sequence" | "dict_pair"
    node_kind: str  # "Tuple" | "List" | "Set" | "Dict"
    lineno: int
    literal_size: int
    value: str
    is_dunder_all: bool
    dict_scope: str  # "" | "module" | "nested"
    literal_index: int
    element_index: int

    def label(self) -> str:
        return (
            f"{TARGET}:{self.lineno} {self.node_kind}[{self.literal_size}] "
            f"element {self.element_index} = {self.value!r}"
        )


def _string_sequences(tree: ast.AST) -> list[ast.AST]:
    """Collection literals whose elements are ALL string constants, in source order.

    EMPTY literals are admitted deliberately and contribute no candidates. They are
    here so that deleting the sole element of a 1-element literal -- TRAP 2's two
    mutants -- leaves a literal the inventory diff can still SEE. Dropping empties
    would make that mutant look like "the literal vanished" and score it invalid.
    """

    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set))
        and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts)
    ]
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def _dicts(tree: ast.AST) -> list[tuple[ast.Dict, str]]:
    """EVERY dict literal, in source order, tagged ``module`` or ``nested``.

    Both scopes are enumerated because the widest PUBLISHED pair (#1225's 213/83) is
    every dict string key at ANY nesting: 81 + 12 + 120 = 213. The module-level subset
    is 28 keys and is a DERIVED universe, not the published one -- the first revision
    of this script had that backwards and said so in its docstring.
    """

    module_level = {
        id(node.value)
        for node in getattr(tree, "body", [])
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(getattr(node, "value", None), ast.Dict)
    }
    found = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)]
    found.sort(key=lambda n: (n.lineno, n.col_offset))
    return [(node, "module" if id(node) in module_level else "nested") for node in found]


def _string_key_positions(node: ast.Dict) -> list[int]:
    return [
        index
        for index, key in enumerate(node.keys)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]


def _inventory_nodes(tree: ast.AST) -> list[ast.AST]:
    """Every literal this script can touch, in the inventory's canonical order."""

    return [*_string_sequences(tree), *[node for node, _ in _dicts(tree)]]


def _row(node: ast.AST) -> tuple[str, tuple[str, ...]]:
    if isinstance(node, ast.Dict):
        return "Dict", tuple(node.keys[i].value for i in _string_key_positions(node))
    return type(node).__name__, tuple(e.value for e in node.elts)


def _inventory(source: str) -> list[tuple[str, tuple[str, ...]]]:
    """Canonical, position-ordered fingerprint of every literal this script can touch.

    The post-mutation validation diffs this, which is what makes "the mutant was never
    applied" and "the mutant changed something else too" both detectable.
    """

    return [_row(node) for node in _inventory_nodes(ast.parse(source))]


def enumerate_candidates(source: str) -> list[Candidate]:
    """Every deletable element in the WIDEST universe, tagged so any universe filters it."""

    tree = ast.parse(source)
    dunder_all = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    dunder_all = node.value

    candidates: list[Candidate] = []
    literal_index = 0
    for node in _string_sequences(tree):
        for element_index, element in enumerate(node.elts):
            candidates.append(
                Candidate(
                    kind="sequence", node_kind=type(node).__name__, lineno=node.lineno,
                    literal_size=len(node.elts), value=element.value,
                    is_dunder_all=node is dunder_all, dict_scope="",
                    literal_index=literal_index, element_index=element_index,
                )
            )
        literal_index += 1
    for node, scope in _dicts(tree):
        positions = _string_key_positions(node)
        for element_index, key_index in enumerate(positions):
            candidates.append(
                Candidate(
                    kind="dict_pair", node_kind="Dict", lineno=node.lineno,
                    literal_size=len(positions), value=node.keys[key_index].value,
                    is_dunder_all=False, dict_scope=scope,
                    literal_index=literal_index, element_index=element_index,
                )
            )
        literal_index += 1
    return candidates


def in_universe(candidate: Candidate, universe: Universe) -> bool:
    if candidate.kind == "dict_pair":
        if universe.dict_key_scope == "none":
            return False
        if universe.dict_key_scope == "module":
            return candidate.dict_scope == "module"
        return True
    if candidate.is_dunder_all:
        return universe.include_dunder_all
    return candidate.literal_size >= universe.min_elements


def _blank(span: str) -> str:
    """Replacement preserving the file's LINE COUNT (see module docstring)."""

    return "\n" * span.count("\n")


def _cut(source: str, start: int, end: int) -> str:
    return source[:start] + _blank(source[start:end]) + source[end:]


def _ensure_trailing_comma(source: str, candidate: Candidate) -> str:
    """Give a 2-element tuple a trailing comma BEFORE one of its elements is cut.

    ``("a", "b")`` minus an element is ``("a")``, which is a plain string, and a
    mutant that changes a TYPE is not the mutant "delete one element". Inserting the
    trailing comma first makes both cuts land on a 1-tuple. It adds no line and
    changes no value, so the inventory diff still sees exactly one deletion.
    """

    tree = ast.parse(source)
    node = _string_sequences(tree)[candidate.literal_index]
    lines = source.splitlines(keepends=True)
    starts = _char_offsets(source)
    end = _offset(lines, starts, node.elts[-1].end_lineno, node.elts[-1].end_col_offset)
    cursor = end
    while cursor < len(source) and source[cursor] in " \t\r\n":
        cursor += 1
    if cursor < len(source) and source[cursor] == ",":
        return source
    return source[:end] + "," + source[end:]


def _dict_entry_spans(source: str, node: ast.Dict, lines: list[str], starts: list[int]) -> list[tuple[int, int]]:
    """Half-open text span of every ENTRY of ``node``, comma-separated, by tokenising.

    A dict pair is NOT ``[key.col_offset, value.end_col_offset)``. A parenthesised or
    bracketed value puts its opening delimiter inside that span and its closing one
    outside, so cutting it leaves a stray ``)`` and the file stops parsing -- measured:
    two candidates at `:2227` failed exactly that way and aborted the first revision's
    widest run. Entry boundaries are therefore the commas at the dict's OWN nesting
    depth, found with ``tokenize`` so a comma inside a string or a nested call cannot
    be mistaken for a separator.
    """

    open_offset = _offset(lines, starts, node.lineno, node.col_offset)
    readline = iter(source.splitlines(keepends=True)).__next__
    depth, spans, entry_start = 0, [], None
    for token in tokenize.generate_tokens(readline):
        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            continue
        offset = _offset(lines, starts, token.start[0], len(lines[token.start[0] - 1][: token.start[1]].encode("utf-8")))
        end_offset = _offset(lines, starts, token.end[0], len(lines[token.end[0] - 1][: token.end[1]].encode("utf-8")))
        if offset < open_offset:
            continue
        if token.string in "([{":
            depth += 1
            if depth == 1:
                entry_start = end_offset
            continue
        if token.string in ")]}":
            depth -= 1
            if depth == 0:
                if entry_start is not None:
                    spans.append((entry_start, offset))
                break
            continue
        if token.string == "," and depth == 1:
            spans.append((entry_start, offset))
            entry_start = end_offset
    trimmed = []
    for start, end in spans:
        text = source[start:end]
        if not text.strip():
            continue  # the trailing comma's empty tail
        lead = len(text) - len(text.lstrip())
        trail = len(text) - len(text.rstrip())
        trimmed.append((start + lead, end - trail))
    return trimmed


def _deleted_subtree_rows(node: ast.Dict, position: int) -> set[int]:
    """ids of every inventory literal that lives INSIDE the pair about to be cut.

    Deleting a ``key: value`` pair legitimately takes the value's own literals with
    it, so the inventory shrinks by more than one row. The first revision scored that
    as "changed the literal INVENTORY, not one element" and reported 7 valid mutants
    as invalid, which is how its widest universe under-counted survivors.
    """

    doomed: set[int] = set()
    for root in (node.keys[position], node.values[position]):
        if root is None:
            continue
        for inner in ast.walk(root):
            if isinstance(inner, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                doomed.add(id(inner))
    return doomed


def apply_deletion(source: str, candidate: Candidate) -> str:
    """Delete ``candidate``'s element and exactly one adjacent comma."""

    if candidate.kind == "sequence" and candidate.node_kind == "Tuple" and candidate.literal_size == 2:
        source = _ensure_trailing_comma(source, candidate)
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    starts = _char_offsets(source)

    if candidate.kind == "sequence":
        node = _string_sequences(tree)[candidate.literal_index]
        element = node.elts[candidate.element_index]
        start = _offset(lines, starts, element.lineno, element.col_offset)
        end = _offset(lines, starts, element.end_lineno, element.end_col_offset)
    else:
        dicts = _dicts(tree)
        node = dicts[candidate.literal_index - len(_string_sequences(tree))][0]
        position = _string_key_positions(node)[candidate.element_index]
        spans = _dict_entry_spans(source, node, lines, starts)
        if len(spans) != len(node.keys):
            raise AssertionError(
                f"{candidate.label()}: tokenised {len(spans)} entries for a dict of "
                f"{len(node.keys)} -- refusing to cut a span I cannot align"
            )
        start, end = spans[position]

    # One adjacent comma: the FOLLOWING one when there is one, else the preceding.
    cursor = end
    while cursor < len(source) and source[cursor] in " \t\r\n":
        cursor += 1
    if cursor < len(source) and source[cursor] == ",":
        mutated = _cut(source, end, cursor + 1)
        return _cut(mutated, start, end)
    back = start - 1
    while back >= 0 and source[back] in " \t\r\n":
        back -= 1
    if back >= 0 and source[back] == ",":
        mutated = _cut(source, start, end)
        return _cut(mutated, back, back + 1)
    if candidate.literal_size == 1:
        # The sole element of a 1-element literal: no comma to take with it, and the
        # result is the legal empty literal. TRAP 2's two mutants land here.
        return _cut(source, start, end)
    raise AssertionError(f"no adjacent comma for {candidate.label()}")


class InvalidMutant(Exception):
    """The applied text is not the intended one-element deletion."""


def _expected_inventory(source: str, candidate: Candidate) -> list[tuple[str, tuple[str, ...]]]:
    """The literal inventory a CORRECT mutant must produce, derived before mutating.

    For a sequence element that is the original inventory with one string dropped from
    one literal. For a dict pair it is that MINUS every literal nested inside the
    deleted pair, because cutting ``"k": {"a": 1}`` legitimately removes the inner dict
    too. Deriving the expectation instead of asserting "exactly one row changed" is
    what lets the widest published universe run at all.
    """

    tree = ast.parse(source)
    nodes = _inventory_nodes(tree)
    target = nodes[candidate.literal_index]
    doomed: set[int] = set()
    if candidate.kind == "dict_pair":
        assert isinstance(target, ast.Dict)
        doomed = _deleted_subtree_rows(target, _string_key_positions(target)[candidate.element_index])

    expected: list[tuple[str, tuple[str, ...]]] = []
    for node in nodes:
        if id(node) in doomed:
            continue
        kind, values = _row(node)
        if node is target:
            values = tuple(
                value for position, value in enumerate(values) if position != candidate.element_index
            )
        expected.append((kind, values))
    return expected


def validated_mutant(source: str, candidate: Candidate) -> str:
    """Apply, then PROVE the mutant is the intended one. Raises InvalidMutant otherwise.

    This is the guard against report 4 section 4.2's "a mutant that was never applied
    scores exactly like a passing test": the mutant must remove exactly one element
    from exactly one literal, keep every other literal identical in VALUE and KIND,
    keep the file parseable, and keep its line count.
    """

    try:
        expected = _expected_inventory(source, candidate)
        mutated = apply_deletion(source, candidate)
        mutated_inventory = _inventory(mutated)
    except SyntaxError as error:
        # A deletion CAN break the grammar. That is an invalid mutant to report, not a
        # crash to abort the sweep with -- the first revision of this script let the
        # SyntaxError escape, so one bad candidate killed the whole run.
        raise InvalidMutant(f"{candidate.label()} does not parse: {error}") from error
    except AssertionError as error:
        raise InvalidMutant(str(error)) from error

    if mutated_inventory != expected:
        extra = [row for row in mutated_inventory if row not in expected]
        missing = [row for row in expected if row not in mutated_inventory]
        raise InvalidMutant(
            f"{candidate.label()} did not produce the expected literal inventory "
            f"({len(mutated_inventory)} rows vs {len(expected)}); unexpected={extra[:2]} "
            f"absent={missing[:2]}"
        )
    if mutated.count("\n") != source.count("\n"):
        raise InvalidMutant(f"{candidate.label()} changed the file's LINE COUNT")
    if mutated == source:
        raise InvalidMutant(f"{candidate.label()} was a no-op")
    return mutated


# --- running -------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def purge_bytecode(root: Path) -> int:
    removed = 0
    for parent in ("src", "tests"):
        for cache in (root / parent).rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
            removed += 1
    return removed


def criterion_paths(targets: tuple[str, ...]) -> list[Path]:
    return [ROOT / (target.replace(".", "/") + ".py") for target in targets]


def criterion_fingerprint(targets: tuple[str, ...]) -> tuple[str, list[str]]:
    """sha256 over the kill criterion's CONTENT, and the per-file digests.

    The command line is not the criterion: two runs of the same command against
    different suite contents are different measurements, and this sweep's whole
    subject is a survivor count published without the thing that produced it.
    """

    return _fingerprint(criterion_paths(targets), "kill-criterion module")


def harness_fingerprint() -> tuple[str, list[str]]:
    """sha256 over the ARM-SELECTION files -- what decides which tree is measured.

    Kept separate from the criterion digest rather than folded into it, because
    ``Publication.criterion_sha256`` is the per-file digest of the criterion module as
    the record carries it, and mixing more files in would make every published pair
    permanently uncomparable.
    """

    return _fingerprint([ROOT / path for path in ARM_SELECTION_FILES], "arm-selection file")


def _fingerprint(paths: list[Path], what: str) -> tuple[str, list[str]]:
    digests, blob = [], b""
    for path in paths:
        if not path.is_file():
            print(f"ERROR: {what} {path} does not exist; refusing to guess", file=sys.stderr)
            # Exit 2 like every other refusal in this script: a caller that tells
            # "refused" from "mismatch" by exit code must not have to special-case one.
            raise SystemExit(2)
        data = path.read_bytes()
        digests.append(f"{path.relative_to(ROOT)}@sha256:{sha256_bytes(data)[:16]}")
        blob += data
    return sha256_bytes(blob), digests


def test_command(targets: tuple[str, ...]) -> list[str]:
    return [sys.executable, "-B", "-m", "unittest", *targets]


def run_tests(root: Path, targets: tuple[str, ...], timeout: float) -> tuple[int, str]:
    environment = dict(os.environ)
    # ABSOLUTE, per-arm PYTHONPATH: report 4 section 4.2 item 4. A relative `src`
    # resolves against a CHILD's cwd and an editable .pth then wins.
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        test_command(targets), cwd=str(root), env=environment,
        capture_output=True, text=True, timeout=timeout,
    )
    return completed.returncode, completed.stdout + completed.stderr


WRONG_CAUSE_MARKERS = (
    "ImportError", "ModuleNotFoundError", "NameError", "SyntaxError",
    "IndentationError", "unittest.loader._FailedTest",
)


def classify(returncode: int, output: str) -> str:
    """kill-by-assertion, kill-by-wrong-cause, or survived."""

    if returncode == 0:
        return "survived"
    for marker in WRONG_CAUSE_MARKERS:
        if marker in output:
            return f"killed_wrong_cause:{marker}"
    if "Ran 0 tests" in output:
        return "killed_wrong_cause:no tests ran"
    return "killed"


@dataclass
class Result:
    candidate: Candidate
    verdict: str
    detail: str = ""


def _first_failure(output: str) -> str:
    for line in output.splitlines():
        if line.startswith(("FAIL: ", "ERROR: ")):
            return line.strip()
    return ""


def whole_worktree_porcelain() -> str:
    """``git status --porcelain`` over the WHOLE worktree, verbatim, for the readout.

    The watched set refuses; this one merely SHOWS. An unwatched edit that changes what
    the sweep measures can then at least be read off the artifact instead of being
    invisible in it.
    """

    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"UNAVAILABLE ({type(error).__name__}: {error})"
    if completed.returncode != 0:
        return f"UNAVAILABLE (git status exited {completed.returncode})"
    return completed.stdout.rstrip() or "(empty -- the whole worktree is clean)"


def worktree_state(paths: list[Path]) -> tuple[bool, str]:
    """(clean, reason). FAILS CLOSED: an unrunnable check is not a clean check.

    The first revision returned "not dirty" from a bare ``except Exception``, so with
    git off the PATH it swept a MODIFIED target and printed a match. An instrument
    that cannot report failure reports success, so every failure mode here refuses.
    Scoped to the kill criterion's files as well as the target: a modified suite is
    just as much a different measurement as a modified target.
    """

    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", *[str(p) for p in paths]],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"git could not be run ({type(error).__name__}: {error})"
    if completed.returncode != 0:
        return False, f"git status exited {completed.returncode}: {completed.stderr.strip()[:200]}"
    if completed.stdout.strip():
        return False, "already modified:\n" + completed.stdout.rstrip()
    return True, "clean"


# --- readout -------------------------------------------------------------------


def stamp(universe: Universe, digest: str, criterion: str, harness: str = "") -> str:
    """Every count is printed through this. That is the whole discipline."""

    return (
        f"[universe={universe.name} threshold=>={universe.min_elements}-element "
        f"dunder_all={'IN' if universe.include_dunder_all else 'OUT'} "
        f"dict_keys={universe.dict_key_scope} "
        f"target@{digest[:16]} criterion@{criterion[:8]} harness@{harness[:8]}]"
    )


def report_universe(
    universe: Universe, results: list[Result], digest: str, criterion: str,
    harness: str, targets: tuple[str, ...], structural_only: bool, worktree: str,
) -> dict:
    selected = [result for result in results if in_universe(result.candidate, universe)]
    deletions = len(selected)
    survivors = [r for r in selected if r.verdict == "survived"]
    wrong_cause = [r for r in selected if r.verdict.startswith("killed_wrong_cause")]
    invalid = [r for r in selected if r.verdict == "invalid"]
    tag = stamp(universe, digest, criterion, harness)

    print(f"{tag} definition: {universe.definition}")
    if universe.note:
        print(f"{tag} note: {universe.note}")
    print(f"{tag} deletions = {deletions}")
    if structural_only:
        print(f"{tag} survivors = NOT MEASURED (--structural-only)")
    else:
        print(f"{tag} survivors = {len(survivors)}   kill criterion: {' '.join(test_command(targets))}")
        print(f"{tag} killed = {deletions - len(survivors) - len(invalid)}"
              f"   of which killed_by_wrong_cause = {len(wrong_cause)}"
              f"   invalid_mutants = {len(invalid)}")
        for result in survivors:
            print(f"{tag} SURVIVOR {result.candidate.label()}")
        for result in wrong_cause:
            print(f"{tag} WRONG CAUSE {result.verdict} {result.candidate.label()} {result.detail}")
        for result in invalid:
            print(f"{tag} INVALID {result.detail}")

    comparable, elsewhere, verdicts, statuses = [], [], [], []
    for publication in universe.publications:
        (comparable if digest.startswith(publication.target_sha256) else elsewhere).append(publication)
    for publication in elsewhere:
        which = KNOWN_TARGET_DIGESTS.get(publication.target_sha256, "an unrecorded file")
        print(f"{tag} NOT COMPARABLE: #{publication.pr or '?'} published "
              f"{publication.deletions} deletions / {publication.survivors} survivors "
              f"[{publication.status}] on {which} (target@{publication.target_sha256}) -- "
              f"{publication.note}")
    for publication in comparable:
        if publication.status != "published":
            print(f"{tag} DISPUTED: {publication.deletions}/{publication.survivors} "
                  f"(#{publication.pr or '?'}, {publication.note}) -- not treated as a result")
            continue
        criterion_note = (
            "same criterion as published"
            if criterion.startswith(publication.criterion_sha256)
            else (f"DIFFERENT criterion: published at {publication.criterion_sha256} "
                  f"({publication.criterion_test_defs} test defs), measured at {criterion[:8]}")
        )
        # The criterion status is folded INTO the verdict, not printed beside it: a
        # consumer scraping `verdict` out of the JSON would otherwise lose the one
        # caveat that says the pair was published against a different suite.
        # `statuses` is the machine-readable half and `verdicts` the prose half. They
        # are separate because folding the criterion caveat into the verdict STRING made
        # the old substring test (`"DIFFER" in verdict`) fire on "DIFFERENT criterion",
        # so three MATCHING universes reported MISMATCH and the exit code went to 1.
        # A status token cannot collide with prose.
        if deletions != publication.deletions:
            statuses.append("deletions_differ")
            verdicts.append(f"DELETIONS DIFFER FROM #{publication.pr} "
                            f"({deletions} vs {publication.deletions}) [{criterion_note}]")
        elif structural_only:
            statuses.append("deletions_match_survivors_unmeasured")
            verdicts.append(f"DELETIONS MATCH #{publication.pr} (survivors not measured) "
                            f"[{criterion_note}]")
        elif len(survivors) != publication.survivors:
            statuses.append("survivors_differ")
            verdicts.append(f"SURVIVORS DIFFER FROM #{publication.pr} "
                            f"({len(survivors)} vs {publication.survivors}) [{criterion_note}]")
        else:
            statuses.append("match")
            verdicts.append(f"MATCHES #{publication.pr} "
                            f"({publication.deletions}/{publication.survivors}) [{criterion_note}]")
        print(f"{tag} vs #{publication.pr} {publication.arm} [{publication.label}]: "
              f"{verdicts[-1]}")

    if not verdicts:
        verdict = "DERIVED -- no comparable publication at this target digest"
        statuses.append("derived")
        print(f"{tag} {verdict}")
    else:
        verdict = "; ".join(verdicts)
    print(f"{tag} worktree: {worktree}")
    print()
    return {
        "universe": universe.name, "definition": universe.definition,
        "min_elements": universe.min_elements,
        "include_dunder_all": universe.include_dunder_all,
        "dict_key_scope": universe.dict_key_scope,
        "derived": universe.derived,
        "target": str(TARGET), "target_sha256": digest,
        "criterion_sha256": criterion,
        "arm_selection_sha256": harness,
        "kill_criterion": None if structural_only else " ".join(test_command(targets)),
        "deletions": deletions,
        "survivors": None if structural_only else len(survivors),
        "survivor_labels": [r.candidate.label() for r in survivors],
        "killed_by_wrong_cause": [r.verdict for r in wrong_cause],
        "invalid_mutants": [r.detail for r in invalid],
        "publications": [
            {"pr": p.pr, "status": p.status, "arm": p.arm, "target_sha256": p.target_sha256,
             "criterion_sha256": p.criterion_sha256, "deletions": p.deletions,
             "survivors": p.survivors, "comparable": digest.startswith(p.target_sha256)}
            for p in universe.publications
        ],
        "verdict": verdict, "comparison_statuses": statuses, "worktree": worktree,
        "whole_worktree_porcelain": whole_worktree_porcelain(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-u", "--universe", action="append", choices=[u.name for u in UNIVERSES],
                        help="universe to report (repeatable; default: all)")
    parser.add_argument("--test-target", action="append",
                        help=f"unittest target forming the kill criterion "
                             f"(default: {' '.join(DEFAULT_TEST_TARGETS)})")
    parser.add_argument("--list", action="store_true", help="list candidates per universe and exit")
    parser.add_argument("--structural-only", action="store_true",
                        help="count deletions only; run no tests")
    parser.add_argument("--timeout", type=float, default=900.0, help="per-mutant test timeout")
    parser.add_argument("--json", type=Path, help="write the readout as JSON here")
    arguments = parser.parse_args(argv)

    universes = [UNIVERSES_BY_NAME[name] for name in (arguments.universe or [u.name for u in UNIVERSES])]
    targets = tuple(arguments.test_target or DEFAULT_TEST_TARGETS)
    target_path = ROOT / TARGET
    original = target_path.read_text(encoding="utf-8")
    digest_before = sha256(target_path)
    criterion_digest, criterion_files = criterion_fingerprint(targets)
    harness_digest, harness_files = harness_fingerprint()

    print(f"# {TARGET} sha256 = {digest_before}")
    known = KNOWN_TARGET_DIGESTS.get(digest_before[:8])
    print(f"# target is {known if known else 'NOT a file any published pair was measured on'}"
          f" -- digest CHECKED against KNOWN_TARGET_DIGESTS, not merely displayed")
    print(f"# kill criterion content sha256 = {criterion_digest}")
    for entry in criterion_files:
        print(f"#   {entry}")
    print(f"# arm-selection (sys.path) content sha256 = {harness_digest}")
    for entry in harness_files:
        print(f"#   {entry}")
    print(f"# repo = {ROOT}")
    print(f"# commit = {_commit()}")
    print(f"# python = {sys.executable}")
    print(f"# kill criterion = {' '.join(test_command(targets))} "
          f"(cwd={ROOT}, PYTHONPATH={ROOT / 'src'})")
    porcelain = whole_worktree_porcelain()
    print("# git status --porcelain (WHOLE worktree, verbatim -- the watched set refuses, "
          "this only shows):")
    for line in porcelain.splitlines():
        print(f"#   {line}")
    print()

    candidates = enumerate_candidates(original)

    if arguments.list:
        for universe in universes:
            tag = stamp(universe, digest_before, criterion_digest, harness_digest)
            selected = [c for c in candidates if in_universe(c, universe)]
            print(f"{tag} deletions = {len(selected)}")
            for candidate in selected:
                print(f"{tag}   {candidate.label()}")
            print()
        threshold_delta = [c for c in candidates if c.kind == "sequence" and c.literal_size == 1]
        print(f"# TRAP 2: threshold >=1 admits exactly {len(threshold_delta)} more mutants than >=2:")
        for candidate in threshold_delta:
            print(f"#   {candidate.label()}")
        return 0

    results: list[Result] = []
    worktree_note = "not checked (--structural-only mutates nothing)"
    if arguments.structural_only:
        results = [Result(candidate=c, verdict="not_measured") for c in candidates]
    else:
        watched = [TARGET, *[p.relative_to(ROOT) for p in criterion_paths(targets)],
                   *ARM_SELECTION_FILES]
        clean, reason = worktree_state(watched)
        if not clean:
            print("ERROR: refusing to sweep -- the watched set (target, kill criterion, "
                  f"arm-selection files) is not verifiably clean: {reason}", file=sys.stderr)
            return 2
        worktree_note = f"verified clean before the sweep: {', '.join(str(p) for p in watched)}"
        purged = purge_bytecode(ROOT)
        print(f"# purged {purged} __pycache__ directories; children run with -B")
        code, output = run_tests(ROOT, targets, arguments.timeout)
        if code != 0:
            print("ERROR: the kill criterion is RED on the unmutated file. Every survivor "
                  "count below the baseline would be meaningless.", file=sys.stderr)
            print(output[-4000:], file=sys.stderr)
            return 2
        print(f"# baseline GREEN: {_ran_line(output)}")
        print()

        needed = [c for c in candidates if any(in_universe(c, u) for u in universes)]
        started = time.time()
        try:
            for position, candidate in enumerate(needed, start=1):
                try:
                    mutated = validated_mutant(original, candidate)
                except InvalidMutant as error:
                    results.append(Result(candidate=candidate, verdict="invalid", detail=str(error)))
                    print(f"# [{position}/{len(needed)}] {'invalid':<28} {candidate.label()}", flush=True)
                    continue
                target_path.write_text(mutated, encoding="utf-8")
                code, output = run_tests(ROOT, targets, arguments.timeout)
                verdict = classify(code, output)
                results.append(Result(candidate=candidate, verdict=verdict, detail=_first_failure(output)))
                print(f"# [{position}/{len(needed)}] {verdict:<28} {candidate.label()}", flush=True)
        finally:
            target_path.write_text(original, encoding="utf-8")
        print(f"# swept {len(needed)} mutants in {time.time() - started:.1f}s")
        digest_after = sha256(target_path)
        print(f"# {TARGET} restored: sha256 = {digest_after} "
              f"({'IDENTICAL' if digest_after == digest_before else 'DIFFERS -- RESTORE FAILED'})")
        if digest_after != digest_before:
            return 2
        print()

    payload = [
        report_universe(universe, results, digest_before, criterion_digest, harness_digest,
                        targets, arguments.structural_only, worktree_note)
        for universe in universes
    ]
    if arguments.json:
        arguments.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"# wrote {arguments.json}")
    bad = [row for row in payload
           if any(status.endswith("_differ") for status in row["comparison_statuses"])]
    for row in bad:
        print(f"# MISMATCH {row['universe']}: {row['verdict']}. A mismatch is a FINDING about this "
              f"file at this digest and this criterion. Do not edit the published pair to agree.")
    return 1 if bad else 0


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - a tarball has no git
        return "unknown"


def _ran_line(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("Ran "):
            return line.strip()
    return "no `Ran N tests` line"


if __name__ == "__main__":
    raise SystemExit(main())
