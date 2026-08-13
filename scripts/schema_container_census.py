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


def _assigned_names(tree: ast.AST) -> dict[int, str]:
    """Map a container node's id() to the variable it is assigned to, where there is one.

    The classification is keyed on the container's NAME as well as its members. Keying on members
    alone let a NEW routing tuple whose member set duplicated an existing row be classified for free
    and then checked by nothing -- which is drill defect #1 (an unregistered routing tuple) reopening
    through the very file added to prevent it. A container with no assignment target (an inline set in
    an `if x not in {...}`) gets "<inline>", which is still distinguishing: two inline containers in
    the same file with the same members are genuinely the same classification question.
    """
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not targets:
            continue
        name = next((t.id for t in targets if isinstance(t, ast.Name)), None)
        if name and isinstance(node.value, (ast.Set, ast.List, ast.Tuple, ast.Dict)):
            out[id(node.value)] = name
    return out


def containers() -> list[tuple[str, int, tuple[str, ...], str]]:
    """(file, line, sorted members, variable name) for every literal container of schema names."""
    found: set[tuple[str, int, tuple[str, ...], str]] = set()
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
            names = _assigned_names(tree)
            for node in ast.walk(tree):
                # Sets/lists/tuples by MEMBER, dicts by KEY. Scanning only the first three left the
                # enumerator's own denominator incomplete, and it cost a full scored run to find out:
                # `OBSERVATION_SCHEMA_CLI_CHOICES` in showdown.py is a dict keyed by the short names,
                # `observation_schema_version_from_choice` raises on an unknown key, and every
                # EngineEnvTest failed on it -- reported as ten surviving conflations. The seventh
                # instance of the class this tool exists to enumerate, missed by the tool because
                # "container" had been defined by three node types rather than by what the code does.
                if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                    elts = node.elts
                elif isinstance(node, ast.Dict):
                    elts = [k for k in node.keys if k is not None]  # `**expr` has a None key
                else:
                    continue
                vals = {
                    e.value for e in elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
                # ALSO containers built from the version CONSTANTS, not just string literals. An
                # earlier version of this docstring asserted those were "covered by the drill's
                # property-tuple mirroring plus PRECONDITION 2's membership check" -- which was FALSE
                # for an ad-hoc set inline in a function body. `export_encoder_tables.py` gates on
                # `{OBSERVATION_SCHEMA_VERSION_V2_2, ..._V3, ..._V4}` in `main()`, nothing mirrored
                # it, and every EngineEnvTest failed on `parser.error("unsupported encoder-table
                # schema")`. Eighth instance of the class, and the second missed because this tool's
                # notion of "container" was narrower than the code's.
                consts = {
                    e.id for e in elts
                    if isinstance(e, ast.Name) and e.id.startswith("OBSERVATION_SCHEMA_VERSION_V")
                }
                # >= 2 members, and EVERY member a schema name in ONE convention. A single name is an
                # ordinary comparison, and a mixed container is not a schema table.
                rel = str(path.relative_to(REPO))
                var = names.get(id(node), "<inline>")
                if len(vals) >= 2 and (vals <= SHORT or vals <= FULL):
                    found.add((rel, node.lineno, tuple(sorted(vals)), var))
                elif len(consts) >= 2 and not vals:
                    found.add((rel, node.lineno, tuple(sorted(consts)), var))
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
        if len(parts) not in (3, 4):
            raise SystemExit(f"census: malformed row (want 3 or 4 fields): {raw!r}")
        kind, path, members = parts[0], parts[1], parts[2]
        var = parts[3] if len(parts) > 3 else "<inline>"
        if kind not in ("REGISTER", "PARTIAL", "MIRRORED"):
            raise SystemExit(f"census: unknown classification {kind!r} in {raw!r}")
        out[(path, tuple(sorted(members.split(","))), var)] = kind
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
    for path, line, members, var in found:
        kind = spec.get((path, members, var))
        if kind is None:
            unclassified.append((path, line, members, var))
        if args.list or kind is None:
            print(f"  {kind or 'UNCLASSIFIED':12} {path}:{line}  {var}  {list(members)}")

    # A classification with no container is as bad as a container with no classification: it means
    # the file records a structure that no longer exists, and the next reader trusts it.
    seen = {(p, m, v) for p, _, m, v in found}
    unused = sorted(k for k in spec if k not in seen)
    if unused:
        print()
        print("  classifications with NO matching container (stale rows):")
        for path, members, var in unused:
            print(f"    {path}  {var}  {list(members)}")

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
