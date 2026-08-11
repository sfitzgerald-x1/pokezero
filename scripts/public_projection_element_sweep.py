"""Per-element mutation sweep over ``src/pokezero/public_projection.py``.

WHY THIS FILE EXISTS AT ALL, and it is the whole point: this sweep has been the
headline deliverable of three merged PRs (#1223, #1225, #1226) and its number was
published FIVE different ways while the generator lived in ``/tmp`` on one machine.
Three agents produced three denominators for "the same" sweep; they reconcile only
because the runs happened to share a box. This program's own rule -- *a digest is
published with the script that produces it, never with a description of it* -- was
earned on a records hash and never applied to counts. **A survivor count is exactly
as unreproducible as a hash when its generator is uncommitted.**

So: the universe and the element threshold are DATA IN THIS FILE, not prose in a PR
body, and every count this script emits is printed next to

  * the universe name and its exact definition,
  * the element threshold,
  * the sha256 of the file that was mutated, and
  * the test command that decided kill-vs-survive.

The two traps that produced the five figures, encoded rather than described
--------------------------------------------------------------------------

TRAP 1 -- **91 collides across two universes on two different trees.** It is the
Class A count at the pre-#1225 file AND the Class A + ``__all__`` count at the
post-#1225 file: same number, different meaning. A bare "91" therefore identifies
nothing. That is why the file digest is printed on every line: (universe,
threshold, digest) is the identifier, and the count is only a value.

TRAP 2 -- **the 93-vs-91 gap is purely an element threshold of >=1 instead of
>=2.** It admits exactly two more mutants, ``("struggle",)`` and
``["attribution_unsafe"]``, both trivial survivors -- so a threshold slip inflates
the deletion count by 2 and the survivor count by 2, which reads like a coverage
regression and is not one. ``--list`` prints the admitted candidates so the delta
is inspectable rather than argued.

The five published universes, all correct for different questions
----------------------------------------------------------------

    universe                          deletions  survivors
    class_a                                  79          1   <- the class #1225 published
    class_a_plus_all                         91         13
    ge1_plus_all                             93         15
    class_a_plus_dict_keys                  107          1
    class_a_plus_all_plus_dict_keys         119         13

``deletions`` is structural (AST only) and is re-derived here on every run.
``survivors`` depends on the kill criterion, which is a COMMAND -- see
``DEFAULT_TEST_TARGETS`` -- and is meaningless without it.

What a mutant is
----------------

One mutant deletes ONE element from ONE collection literal (or one ``key: value``
pair from one module-level dict literal) and runs the kill criterion. Deletion is
surgical text removal that PRESERVES THE LINE COUNT of the file (removed spans are
replaced by their own newlines), because ``public_projection.py`` and its
neighbours are cited by line elsewhere in the tree and a mutant that dies of a
citation gate has died of the wrong cause.

Three hazards from report 4 section 4.2 / 4.4 are handled in code, not in a README:

* **A mutant that was never applied scores as a kill.** Every mutant is re-parsed
  and diffed against the original literal inventory before the tests run; a mutant
  that does not remove exactly the intended element (for example a 2-tuple
  degenerating into a plain string) is reported as ``invalid``, never as a kill.
* **Stale bytecode.** ``__pycache__`` under ``src/`` and ``tests/`` is purged once
  at start and every child runs ``python -B``.
* **A mutant dying of the wrong cause.** Deaths are classified from the child's
  output and a per-cause histogram is printed; a kill whose traceback is an
  ``ImportError``/``NameError``/``SyntaxError``/collection error is reported
  separately from a kill by assertion.

Usage
-----

    python scripts/public_projection_element_sweep.py --list
    python scripts/public_projection_element_sweep.py                # all universes
    python scripts/public_projection_element_sweep.py -u class_a
    python scripts/public_projection_element_sweep.py --structural-only

The sweep mutates the working tree in place and restores it byte-for-byte,
printing the sha256 before and after; it refuses to start if the target file is
already dirty.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The file under mutation. Every figure this script prints is about THIS file at
#: the sha256 printed with it, and about nothing else.
TARGET = Path("src/pokezero/public_projection.py")

#: The kill criterion. It is a command, and it is published with every count.
#: `tests.test_public_projection` is the module that owns this file; a wider
#: selection is a DIFFERENT universe of the same name and must be stated as such
#: (pass `--test-target` and read it back off the readout).
DEFAULT_TEST_TARGETS = ("tests.test_public_projection",)


@dataclass(frozen=True)
class Universe:
    """A named question. The counts are only meaningful attached to one of these."""

    name: str
    #: Minimum number of elements a collection literal must have to be admitted.
    #: >=2 is Class A. >=1 admits the two single-element literals and is TRAP 2.
    min_elements: int
    #: Whether the ``__all__`` export list is admitted. TRAP 1 lives here.
    include_dunder_all: bool
    #: Whether ``key: value`` pairs of module-level dict literals are admitted.
    include_module_dict_keys: bool
    #: The count recorded when this universe was published. NOT a target to tune
    #: the script to: a mismatch is a finding about the file (read the digest),
    #: not a licence to edit this number.
    recorded_deletions: int
    recorded_survivors: int
    note: str


UNIVERSES: tuple[Universe, ...] = (
    Universe(
        name="class_a",
        min_elements=2,
        include_dunder_all=False,
        include_module_dict_keys=False,
        recorded_deletions=79,
        recorded_survivors=1,
        note="the class #1225 published: >=2-element string-literal collections, __all__ EXCLUDED",
    ),
    Universe(
        name="class_a_plus_all",
        min_elements=2,
        include_dunder_all=True,
        include_module_dict_keys=False,
        recorded_deletions=91,
        recorded_survivors=13,
        note="TRAP 1: 91 is ALSO Class A at the pre-#1225 file. Same number, different meaning",
    ),
    Universe(
        name="ge1_plus_all",
        min_elements=1,
        include_dunder_all=True,
        include_module_dict_keys=False,
        recorded_deletions=93,
        recorded_survivors=15,
        note="TRAP 2: 93 - 91 is a THRESHOLD slip, admitting ('struggle',) and ['attribution_unsafe']",
    ),
    Universe(
        name="class_a_plus_dict_keys",
        min_elements=2,
        include_dunder_all=False,
        include_module_dict_keys=True,
        recorded_deletions=107,
        recorded_survivors=1,
        note="Class A plus module-level dict key/value pairs",
    ),
    Universe(
        name="class_a_plus_all_plus_dict_keys",
        min_elements=2,
        include_dunder_all=True,
        include_module_dict_keys=True,
        recorded_deletions=119,
        recorded_survivors=13,
        note="the widest universe published",
    ),
)

UNIVERSES_BY_NAME = {universe.name: universe for universe in UNIVERSES}


# --- source surgery -----------------------------------------------------------


def _char_offsets(source: str) -> list[int]:
    """Start offset of every 1-based line, in CHARACTERS."""

    offsets = [0, 0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _offset(lines: list[str], starts: list[int], lineno: int, col: int) -> int:
    """Absolute character offset of an ``ast`` (lineno, col_offset) pair.

    ``col_offset`` is a UTF-8 BYTE offset, and this file contains non-ASCII prose,
    so the prefix is decoded rather than sliced.
    """

    prefix = lines[lineno - 1].encode("utf-8")[:col].decode("utf-8")
    return starts[lineno] + len(prefix)


@dataclass(frozen=True)
class Candidate:
    """One deletable element."""

    index: int
    kind: str  # "sequence" | "dict_pair"
    node_kind: str  # "Tuple" | "List" | "Set" | "Dict"
    lineno: int
    literal_size: int
    value: str
    is_dunder_all: bool
    is_module_dict_key: bool
    #: the literal's identity within the inventory, for post-mutation validation
    literal_index: int
    element_index: int

    def label(self) -> str:
        return f"{TARGET}:{self.lineno} {self.node_kind}[{self.literal_size}] element {self.element_index} = {self.value!r}"


def _string_sequences(tree: ast.AST) -> list[ast.AST]:
    """Collection literals whose elements are ALL string constants, in source order.

    EMPTY literals are admitted deliberately, and they contribute no candidates.
    They are here so that deleting the sole element of a 1-element literal -- TRAP
    2's ``("struggle",)`` and ``["attribution_unsafe"]`` -- leaves a literal the
    inventory diff can still SEE. Dropping empties would make that mutant look like
    "the literal vanished" and score it invalid, which understates the >=1 universe
    by exactly those two.
    """

    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
                found.append(node)
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def _module_dicts(tree: ast.Module) -> list[ast.Dict]:
    """Dict literals bound by a MODULE-LEVEL assignment, in source order.

    Module level, not every dict in the file: a dict built inside a function body
    is a different object (its keys are usually protocol tags being dispatched on,
    not a table the module exports), and the published 107/119 figures are the
    module-level ones. Stated here so the next reader does not have to re-derive
    which 28 keys those were.
    """

    found = []
    for node in tree.body:
        value = getattr(node, "value", None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Dict):
            found.append(value)
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def _inventory(source: str) -> list[tuple[str, int, tuple[str, ...]]]:
    """Canonical, position-ordered fingerprint of every literal this script can touch.

    The post-mutation validation diffs this, which is what makes "the mutant was
    never applied" and "the mutant changed something else too" both detectable.
    """

    tree = ast.parse(source)
    rows: list[tuple[str, int, tuple[str, ...]]] = []
    for node in _string_sequences(tree):
        rows.append((type(node).__name__, node.lineno, tuple(e.value for e in node.elts)))
    for node in _module_dicts(tree):
        rows.append(
            (
                "Dict",
                node.lineno,
                tuple(k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)),
            )
        )
    return rows


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
                    index=len(candidates),
                    kind="sequence",
                    node_kind=type(node).__name__,
                    lineno=node.lineno,
                    literal_size=len(node.elts),
                    value=element.value,
                    is_dunder_all=node is dunder_all,
                    is_module_dict_key=False,
                    literal_index=literal_index,
                    element_index=element_index,
                )
            )
        literal_index += 1
    for node in _module_dicts(tree):
        pairs = [
            index
            for index, key in enumerate(node.keys)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        for element_index, key_index in enumerate(pairs):
            candidates.append(
                Candidate(
                    index=len(candidates),
                    kind="dict_pair",
                    node_kind="Dict",
                    lineno=node.lineno,
                    literal_size=len(pairs),
                    value=node.keys[key_index].value,
                    is_dunder_all=False,
                    is_module_dict_key=True,
                    literal_index=literal_index,
                    element_index=element_index,
                )
            )
        literal_index += 1
    return candidates


def in_universe(candidate: Candidate, universe: Universe) -> bool:
    if candidate.is_module_dict_key:
        return universe.include_module_dict_keys
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
    mutant that changes a TYPE is not the mutant "delete one element". Inserting
    the trailing comma first makes both cuts land on a 1-tuple instead. It adds no
    line and changes no value, so the inventory diff in ``validated_mutant`` still
    sees exactly one deletion.
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
        remaining = len(node.elts) - 1
    else:
        dicts = _module_dicts(tree)
        node = dicts[candidate.literal_index - len(_string_sequences(tree))]
        keys = [
            index
            for index, key in enumerate(node.keys)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        key = node.keys[keys[candidate.element_index]]
        value = node.values[keys[candidate.element_index]]
        start = _offset(lines, starts, key.lineno, key.col_offset)
        end = _offset(lines, starts, value.end_lineno, value.end_col_offset)
        remaining = len(keys) - 1

    # One adjacent comma: the FOLLOWING one when there is one, else the preceding.
    cursor = end
    while cursor < len(source) and source[cursor] in " \t\r\n":
        cursor += 1
    if cursor < len(source) and source[cursor] == ",":
        mutated = _cut(source, end, cursor + 1)
        mutated = _cut(mutated, start, end)
    else:
        back = start - 1
        while back >= 0 and source[back] in " \t\r\n":
            back -= 1
        if back >= 0 and source[back] == ",":
            mutated = _cut(source, start, end)
            mutated = _cut(mutated, back, back + 1)
        elif candidate.literal_size == 1:
            # The sole element of a 1-element literal: there is no comma to take with
            # it, and the result is the legal empty literal. TRAP 2's two mutants.
            mutated = _cut(source, start, end)
        else:
            raise AssertionError(f"no adjacent comma for {candidate.label()}")

    return mutated


def validated_mutant(source: str, candidate: Candidate) -> str:
    """Apply, then PROVE the mutant is the intended one. Raises otherwise.

    This is the guard against report 4 section 4.2's "a mutant that was never
    applied scores exactly like a passing test": the mutant must remove exactly one
    element from exactly one literal, keep every other literal byte-identical in
    VALUE and in KIND, and keep the file's line count.
    """

    original_inventory = _inventory(source)
    mutated = apply_deletion(source, candidate)
    mutated_inventory = _inventory(mutated)
    if len(mutated_inventory) != len(original_inventory):
        raise AssertionError(f"{candidate.label()} changed the literal INVENTORY, not one element")
    for index, (before, after) in enumerate(zip(original_inventory, mutated_inventory)):
        if index == candidate.literal_index:
            expected = tuple(
                value for position, value in enumerate(before[2]) if position != candidate.element_index
            )
            if after[0] != before[0] or after[2] != expected:
                raise AssertionError(
                    f"{candidate.label()} produced {after[0]}{after[2]!r}, expected {before[0]}{expected!r}"
                )
        elif after[0] != before[0] or after[2] != before[2]:
            raise AssertionError(f"{candidate.label()} also changed the literal at line {before[1]}")
    if mutated.count("\n") != source.count("\n"):
        raise AssertionError(f"{candidate.label()} changed the file's LINE COUNT")
    if mutated == source:
        raise AssertionError(f"{candidate.label()} was a no-op")
    return mutated


# --- running -------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def purge_bytecode(root: Path) -> int:
    removed = 0
    for parent in ("src", "tests"):
        for cache in (root / parent).rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
            removed += 1
    return removed


def test_command(targets: tuple[str, ...]) -> list[str]:
    return [sys.executable, "-B", "-m", "unittest", *targets]


def run_tests(root: Path, targets: tuple[str, ...], timeout: float) -> tuple[int, str]:
    environment = dict(os.environ)
    # ABSOLUTE, per-arm PYTHONPATH: report 4 section 4.2 item 4. A relative `src`
    # resolves against a CHILD's cwd and an editable .pth then wins.
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        test_command(targets),
        cwd=str(root),
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout + completed.stderr


WRONG_CAUSE_MARKERS = (
    "ImportError",
    "ModuleNotFoundError",
    "NameError",
    "SyntaxError",
    "IndentationError",
    "unittest.loader._FailedTest",
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
    first_failure: str = ""


def _first_failure(output: str) -> str:
    for line in output.splitlines():
        if line.startswith(("FAIL: ", "ERROR: ")):
            return line.strip()
    return ""


# --- readout -------------------------------------------------------------------


def stamp(universe: Universe, digest: str) -> str:
    """Every count is printed through this. That is the whole discipline."""

    return (
        f"[universe={universe.name} threshold=>={universe.min_elements}-element "
        f"dunder_all={'IN' if universe.include_dunder_all else 'OUT'} "
        f"module_dict_keys={'IN' if universe.include_module_dict_keys else 'OUT'} "
        f"{TARGET}@sha256:{digest[:16]}]"
    )


def report_universe(universe: Universe, results: list[Result], digest: str, targets, structural_only: bool) -> dict:
    selected = [result for result in results if in_universe(result.candidate, universe)]
    deletions = len(selected)
    survivors = [r for r in selected if r.verdict == "survived"]
    wrong_cause = [r for r in selected if r.verdict.startswith("killed_wrong_cause")]
    invalid = [r for r in selected if r.verdict == "invalid"]
    tag = stamp(universe, digest)

    print(f"{tag} {universe.note}")
    print(f"{tag} deletions = {deletions}   (recorded {universe.recorded_deletions})")
    if structural_only:
        print(f"{tag} survivors = NOT MEASURED (--structural-only)")
    else:
        print(
            f"{tag} survivors = {len(survivors)}   (recorded {universe.recorded_survivors})"
            f"   kill criterion: {' '.join(test_command(tuple(targets)))}"
        )
        print(f"{tag} killed = {deletions - len(survivors) - len(invalid)}"
              f"   of which killed_by_wrong_cause = {len(wrong_cause)}   invalid_mutants = {len(invalid)}")
        for result in survivors:
            print(f"{tag} SURVIVOR {result.candidate.label()}")
        for result in wrong_cause:
            print(f"{tag} WRONG CAUSE {result.verdict} {result.candidate.label()} {result.first_failure}")
        for result in invalid:
            print(f"{tag} INVALID {result.candidate.label()} {result.first_failure}")

    if deletions != universe.recorded_deletions:
        verdict = "DELETIONS DIFFER FROM RECORDED"
    elif structural_only:
        verdict = "DELETIONS MATCH RECORDED (survivors not measured)"
    elif len(survivors) == universe.recorded_survivors:
        verdict = "MATCHES RECORDED"
    else:
        verdict = "DELETIONS MATCH, SURVIVORS DIFFER FROM RECORDED"
    print(f"{tag} {verdict}")
    if not verdict.startswith(("MATCHES RECORDED", "DELETIONS MATCH RECORDED")):
        print(
            f"{tag} NOTE: a mismatch is a FINDING about this file at this digest and this "
            "kill criterion. Do not edit the recorded numbers to agree with it."
        )
    print()
    return {
        "universe": universe.name,
        "min_elements": universe.min_elements,
        "include_dunder_all": universe.include_dunder_all,
        "include_module_dict_keys": universe.include_module_dict_keys,
        "target": str(TARGET),
        "target_sha256": digest,
        "kill_criterion": None if structural_only else " ".join(test_command(tuple(targets))),
        "deletions": deletions,
        "recorded_deletions": universe.recorded_deletions,
        "survivors": None if structural_only else len(survivors),
        "recorded_survivors": universe.recorded_survivors,
        "survivor_labels": [r.candidate.label() for r in survivors],
        "killed_by_wrong_cause": [r.verdict for r in wrong_cause],
        "invalid_mutants": [r.candidate.label() for r in invalid],
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-u",
        "--universe",
        action="append",
        choices=[u.name for u in UNIVERSES],
        help="universe to report (repeatable; default: all five published universes)",
    )
    parser.add_argument(
        "--test-target",
        action="append",
        help=f"unittest target forming the kill criterion (default: {' '.join(DEFAULT_TEST_TARGETS)})",
    )
    parser.add_argument("--list", action="store_true", help="list candidates per universe and exit")
    parser.add_argument(
        "--structural-only", action="store_true", help="count deletions only; run no tests"
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="per-mutant test timeout")
    parser.add_argument("--json", type=Path, help="write the readout as JSON here")
    arguments = parser.parse_args(argv)

    universes = [UNIVERSES_BY_NAME[name] for name in (arguments.universe or [u.name for u in UNIVERSES])]
    targets = tuple(arguments.test_target or DEFAULT_TEST_TARGETS)
    target_path = ROOT / TARGET
    original = target_path.read_text(encoding="utf-8")
    digest_before = sha256(target_path)

    print(f"# {TARGET} sha256 = {digest_before}")
    print(f"# repo = {ROOT}")
    print(f"# commit = {_commit()}")
    print(f"# python = {sys.executable}")
    print(f"# kill criterion = {' '.join(test_command(targets))} (cwd={ROOT}, PYTHONPATH={ROOT / 'src'})")
    print()

    candidates = enumerate_candidates(original)

    if arguments.list:
        for universe in universes:
            tag = stamp(universe, digest_before)
            selected = [c for c in candidates if in_universe(c, universe)]
            print(f"{tag} deletions = {len(selected)}   (recorded {universe.recorded_deletions})")
            for candidate in selected:
                print(f"{tag}   {candidate.label()}")
            print()
        # TRAP 2, printed rather than described.
        threshold_delta = [c for c in candidates if not c.is_module_dict_key and c.literal_size == 1]
        print(f"# threshold >=1 admits exactly {len(threshold_delta)} more mutants than >=2:")
        for candidate in threshold_delta:
            print(f"#   {candidate.label()}")
        return 0

    results: list[Result] = []
    if arguments.structural_only:
        results = [Result(candidate=c, verdict="not_measured") for c in candidates]
    else:
        purged = purge_bytecode(ROOT)
        print(f"# purged {purged} __pycache__ directories; children run with -B")
        if _dirty(TARGET):
            print(f"ERROR: {TARGET} is already modified; refusing to sweep a dirty tree", file=sys.stderr)
            return 2
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
                except AssertionError as error:
                    results.append(Result(candidate=candidate, verdict="invalid", first_failure=str(error)))
                    continue
                target_path.write_text(mutated, encoding="utf-8")
                code, output = run_tests(ROOT, targets, arguments.timeout)
                verdict = classify(code, output)
                results.append(
                    Result(candidate=candidate, verdict=verdict, first_failure=_first_failure(output))
                )
                print(
                    f"# [{position}/{len(needed)}] {verdict:<28} {candidate.label()}",
                    flush=True,
                )
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
        report_universe(universe, results, digest_before, targets, arguments.structural_only)
        for universe in universes
    ]
    if arguments.json:
        arguments.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"# wrote {arguments.json}")
    return 0 if all(row["verdict"].startswith(("MATCHES RECORDED", "DELETIONS MATCH RECORDED")) for row in payload) else 1


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - a tarball export has no git
        return "unknown"


def _dirty(path: Path) -> bool:
    try:
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:  # pragma: no cover
        return False
    return bool(status.strip())


def _ran_line(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("Ran "):
            return line.strip()
    return "no `Ran N tests` line"


if __name__ == "__main__":
    raise SystemExit(main())
