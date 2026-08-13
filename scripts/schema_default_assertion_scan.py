#!/usr/bin/env python3
"""Find the tests whose ASSERTIONS read the global schema default, and split them by kind.

This is the tool behind the rotation drill's rubric
(`tests/data/schema_drill_expected_breakages.txt`). That file is the drill's pass condition --
the breakage set must equal it exactly -- so "how was this set derived" has to be answerable by
a command rather than by prose. This is that command.

Three buckets, and the distinction is the whole point:

  PIN    a test whose assertions read the process default and NOTHING else. It legitimately
         ANSWERS "which schema does a fresh artifact get", so a rotation MUST break it. These
         are the class-(iii) rows in the rubric.

  MIXED  a test that reads the default AND asserts something a rotation does not change
         (membership in SUPPORTED_*, a spec's width, a layout fact). It breaks under a rotation
         for either reason, so deleting its default read leaves it still breaking -- which
         blinds the drill's EXPECTED-BUT-DID-NOT-BREAK detector, the only guard against a pin
         silently going stale. Every MIXED test is a defect to split (drill defect D2).

  OTHER  a test that merely USES the default to build something. Not class (iii); it is an
         unmigrated site and belongs to the ledger's census, not to this rubric.

The PIN/MIXED split is syntactic and deliberately conservative: an assertion counts as a
default read if any operand mentions OBSERVATION_SCHEMA_VERSION or
DEFAULT_REPLAY_OBSERVATION_SPEC as a bare name. A test is a PIN only if EVERY one of its
assertions is a default read. That can misfile a test whose non-default assertion happens to be
rotation-invariant; it cannot silently call a MIXED test a PIN, which is the direction that
would matter.

WHAT THIS DOES NOT DECIDE. A PIN still has to be asserted POSITIVELY (`default IS v2.2`) to
belong in the rubric. The negative form (`v4 is not the default`) survives a rotation -- under a
rotation to a synthetic v5-drill, v4 is still not the default -- so it never breaks, and a
non-breaking row in the rubric would report a missing breakage forever. Polarity is not
inferable from the assertion alone (`assertEqual(DEFAULT, V2_2)` and
`assertNotEqual(DEFAULT, V4)` are both `assertEqual`-shaped reads of the same global), so this
tool reports the assertion text and the rubric records the judgement. See the rubric's D2 note.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

# The two module-level globals that hold "whatever nobody chose". A test asserting on either is
# reading the mutable default; a test asserting on OBSERVATION_SCHEMA_VERSION_V2_2 (a specific
# version) is not, which is why these are matched as whole bare names.
DEFAULT_GLOBALS = ("OBSERVATION_SCHEMA_VERSION", "DEFAULT_REPLAY_OBSERVATION_SPEC")


def _reads_default(call: ast.Call) -> bool:
    """True if any operand of this assertion mentions a default global as a bare Name.

    Matched on the AST, not on the unparsed text: `OBSERVATION_SCHEMA_VERSION_V2_2` contains
    `OBSERVATION_SCHEMA_VERSION` as a substring but is a different, immutable name, and a
    substring match would file every version-specific assertion as a default read.
    """
    for node in ast.walk(call):
        if isinstance(node, ast.Name) and node.id in DEFAULT_GLOBALS:
            return True
        # `observation.OBSERVATION_SCHEMA_VERSION` / `obs.DEFAULT_REPLAY_OBSERVATION_SPEC`
        if isinstance(node, ast.Attribute) and node.attr in DEFAULT_GLOBALS:
            return True
    return False


def _assertions(fn: ast.AST) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr.startswith("assert")
    ]


def scan() -> tuple[list[dict], int]:
    """Return (rows, files_scanned). One row per test method holding >=1 default-read assertion."""
    rows: list[dict] = []
    files = sorted(TESTS.rglob("test_*.py"))
    for path in files:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in cls.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not fn.name.startswith("test"):
                    continue
                asserts = _assertions(fn)
                if not asserts:
                    continue
                reads = [a for a in asserts if _reads_default(a)]
                if not reads:
                    continue
                kind = "PIN" if len(reads) == len(asserts) else "MIXED"
                rows.append(
                    {
                        "id": f"{path.relative_to(REPO)}::{cls.name}::{fn.name}",
                        "kind": kind,
                        "line": fn.lineno,
                        "default_asserts": len(reads),
                        "total_asserts": len(asserts),
                        "texts": [ast.unparse(a) for a in reads],
                    }
                )
    return rows, len(files)


def _self_test() -> int:
    """Prove the three buckets bind, and that the version-specific name is NOT a default read."""
    cases = [
        ("self.assertEqual(OBSERVATION_SCHEMA_VERSION, X)", True),
        ("self.assertEqual(DEFAULT_REPLAY_OBSERVATION_SPEC, X)", True),
        ("self.assertEqual(observation.OBSERVATION_SCHEMA_VERSION, X)", True),
        ("self.assertIn(OBSERVATION_SCHEMA_VERSION, TABLE)", True),
        ("self.assertEqual(spec.schema_version, OBSERVATION_SCHEMA_VERSION)", True),
        # The near-miss that a substring match would get wrong, in both spellings.
        ("self.assertEqual(OBSERVATION_SCHEMA_VERSION_V2_2, X)", False),
        ("self.assertEqual(observation.OBSERVATION_SCHEMA_VERSION_V4, X)", False),
        ("self.assertEqual(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS, X)", False),
        ("self.assertEqual(spec.numeric_feature_count, 132)", False),
    ]
    bad = 0
    for src, want in cases:
        call = ast.parse(src).body[0].value
        got = _reads_default(call)
        if got != want:
            print(f"  FAIL want={want} got={got}: {src}")
            bad += 1
    print(f"  _reads_default: {len(cases) - bad}/{len(cases)} probes correct")

    # Bucket classification, end to end, on synthetic source.
    mod = ast.parse(
        "class T:\n"
        "    def test_pin(self):\n"
        "        self.assertEqual(OBSERVATION_SCHEMA_VERSION, 'x')\n"
        "    def test_mixed(self):\n"
        "        self.assertEqual(OBSERVATION_SCHEMA_VERSION, 'x')\n"
        "        self.assertEqual(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS, ())\n"
        "    def test_other(self):\n"
        "        self.assertEqual(1, 1)\n"
    )
    fns = {f.name: f for f in mod.body[0].body}
    expect = {"test_pin": "PIN", "test_mixed": "MIXED", "test_other": None}
    for name, want_kind in expect.items():
        asserts = _assertions(fns[name])
        reads = [a for a in asserts if _reads_default(a)]
        got_kind = None if not reads else ("PIN" if len(reads) == len(asserts) else "MIXED")
        status = "ok" if got_kind == want_kind else "FAIL"
        if status == "FAIL":
            bad += 1
        print(f"  {status}: {name} -> {got_kind} (want {want_kind})")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mixed", action="store_true", help="only MIXED (the D2 defect class)")
    ap.add_argument("--pins", action="store_true", help="only PIN (class-(iii) candidates)")
    ap.add_argument("--texts", action="store_true", help="print the matched assertion source")
    ap.add_argument("--self-test", action="store_true", help="prove the matcher binds; exit 1 on failure")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    rows, files_scanned = scan()
    wanted = rows
    if args.mixed:
        wanted = [r for r in rows if r["kind"] == "MIXED"]
    elif args.pins:
        wanted = [r for r in rows if r["kind"] == "PIN"]

    pins = sum(1 for r in rows if r["kind"] == "PIN")
    mixed = len(rows) - pins
    # Denominator first: every count below is out of these.
    print(f"test files scanned:            {files_scanned}")
    print(f"tests asserting on the default: {len(rows)}   (PIN {pins} / MIXED {mixed})")
    print(f"shown:                          {len(wanted)}")
    print()
    for r in sorted(wanted, key=lambda r: (r["kind"], r["id"])):
        print(f"{r['kind']:5} {r['id']}  L{r['line']}  {r['default_asserts']}/{r['total_asserts']} asserts")
        if args.texts:
            for t in r["texts"]:
                print(f"          {t}")
    if mixed and (args.mixed or not (args.pins or args.mixed)):
        print()
        print(
            f"NOTE: {mixed} MIXED test(s) remain. Each one blinds the drill's dead-pin detector "
            "for its own default read (D2); split it into a clean pin plus a residual that holds "
            "no default read."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
