#!/usr/bin/env python3
"""Enumerate every literal container of schema NAMES in src/ and scripts/, and classify it.

The rotation drill injects a synthetic schema and then asserts that whatever breaks broke because
the DEFAULT MOVED. That claim is only as good as the registration: if any structure keyed by schema
name does not know about the synthetic schema, its consumers fail on the drill's own omission and
get reported as surviving defects.

Six of those were found the expensive way -- run the full drill (17-70 minutes), read a stack trace,
add the one table it named, repeat. SUPPORTED, GROUPED_LAYOUT, FEATURE_PACK, V2_1_LINEAGE, the two
census maps, `_EXPORTABLE_TABLE_SCHEMAS`, and the exporter CLI's argparse `choices`. That loop does
not converge, because after each fix the instrument still had no way to say what it had missed; a
clean run only ever proved that the tables I had thought of were registered.

So the containers are ENUMERATED from the AST and matched against a committed classification
(`tests/data/schema_drill_schema_containers.txt`). An unclassified container is a hard abort, before
any verdict. Two classifications, and the distinction is the point:

  REGISTER  the container means "every supported schema", so the synthetic schema belongs in it.
  PARTIAL   a deliberately chosen subset. `vocab_shift_probe` compares one named PAIR of schemas;
            adding a third does not widen that comparison, it changes what the script does.

WHAT THIS DOES NOT CLAIM. It finds containers whose members are string LITERALS. A container built
from the version CONSTANTS (`(OBSERVATION_SCHEMA_VERSION_V3, ...)`) is a different shape and is
covered by the drill's property-tuple mirroring plus PRECONDITION 2's membership check. Neither
mechanism sees a set assembled at runtime from a comprehension; nothing here pretends otherwise.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SPEC = REPO / "tests" / "data" / "schema_drill_schema_containers.txt"

SHORT = {"v2", "v2.1", "v2.2", "v3", "v4"}
FULL = {f"pokezero.observation.{s}" for s in SHORT}


def containers() -> list[tuple[str, int, tuple[str, ...]]]:
    """(file, line, sorted members) for every literal set/list/tuple of schema names."""
    found: set[tuple[str, int, tuple[str, ...]]] = set()
    roots = [REPO / "src", REPO / "scripts"]
    # This file is the INSTRUMENT, not the subject: its SHORT/FULL sets are the vocabulary it scans
    # WITH, and scanning them would report the measuring device as a table needing registration.
    # Excluded by resolved path rather than by name, so a copy under a different name is still
    # scanned -- only the running census is skipped.
    self_path = pathlib.Path(__file__).resolve()
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if path.resolve() == self_path:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                    continue
                vals = {
                    e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
                # >= 2 members, and EVERY member a schema name in one naming convention. A single
                # name is an ordinary string comparison, and a mixed container is not a schema table.
                if len(vals) >= 2 and (vals <= SHORT or vals <= FULL):
                    rel = str(path.relative_to(REPO))
                    found.add((rel, node.lineno, tuple(sorted(vals))))
    return sorted(found)


def classification() -> dict[tuple[str, tuple[str, ...]], str]:
    """Committed classification, keyed on (file, members) -- NOT on line, which drifts."""
    if not SPEC.is_file():
        raise SystemExit(f"census: no classification file at {SPEC}")
    out: dict[tuple[str, tuple[str, ...]], str] = {}
    for raw in SPEC.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise SystemExit(f"census: malformed row (want 3 fields): {raw!r}")
        kind, path, members = parts
        if kind not in ("REGISTER", "PARTIAL"):
            raise SystemExit(f"census: unknown classification {kind!r} in {raw!r}")
        out[(path, tuple(sorted(members.split(","))))] = kind
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print every container and its class")
    ap.add_argument(
        "--check", action="store_true",
        help="exit 15 if any container is unclassified, or any classification is now unused",
    )
    args = ap.parse_args()

    found = containers()
    spec = classification()
    print(f"schema-name literal containers in src/ and scripts/: {len(found)}")
    print(f"classified in {SPEC.relative_to(REPO)}:                {len(spec)}")
    print()

    unclassified = []
    for path, line, members in found:
        kind = spec.get((path, members))
        if kind is None:
            unclassified.append((path, line, members))
        if args.list or kind is None:
            print(f"  {kind or 'UNCLASSIFIED':12} {path}:{line}  {list(members)}")

    # A classification with no container is as bad as a container with no classification: it means
    # the file records a structure that no longer exists, and the next reader trusts it.
    seen = {(p, m) for p, _, m in found}
    unused = sorted(k for k in spec if k not in seen)
    if unused:
        print()
        print("  classifications with NO matching container (stale rows):")
        for path, members in unused:
            print(f"    {path}  {list(members)}")

    if not args.check:
        return 0
    if unclassified:
        print()
        print(f"ABORT: {len(unclassified)} schema-name container(s) are not classified. The drill "
              "cannot know whether to register its synthetic schema in them, and an unregistered "
              "container makes its consumers fail on the drill's own omission -- which has happened "
              "six times and been reported as surviving defects each time.")
        print(f"Classify each in {SPEC.relative_to(REPO)} as REGISTER or PARTIAL.")
        return 15
    if unused:
        print()
        print(f"ABORT: {len(unused)} classification row(s) match no container in the tree. Remove "
              "them; a stale row is a claim about the code that is no longer true.")
        return 15
    print("  every schema-name container is classified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
