#!/usr/bin/env python3
"""Find schema IDENTITY gates in src/ -- comparisons against a NAMED schema version.

An identity gate asks "is this schema exactly v<n>?" where it almost always means "does this
schema have property P". The property form picks up every future schema with P; the identity form
silently excludes them and takes the wrong branch. Six such gates were found and converted during
the v4 default rotation; two more evaded the first guard because it was a line-oriented grep for
`== OBSERVATION_SCHEMA_VERSION_V<n>` and missed `!=`, `in (...)`, string literals, dict keys,
`is`, `match/case`, and multi-line wraps.

This scan is AST-based, so the syntactic form does not matter -- only that a comparison names a
version. Deliberate requirements ("this capture REQUIRES v3") are legitimate; they live in
ALLOWLIST with a reason, so a new gate is an authorship-time decision rather than a silent one.

Usage:  python3 scripts/schema_identity_gate_scan.py [--json]
Exit 0 = no unlisted identity gates. Exit 1 = at least one; each is printed.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERSION_NAMES = {
    "OBSERVATION_SCHEMA_VERSION_V2",
    "OBSERVATION_SCHEMA_VERSION_V2_1",
    "OBSERVATION_SCHEMA_VERSION_V2_2",
    "OBSERVATION_SCHEMA_VERSION_V3",
    "OBSERVATION_SCHEMA_VERSION_V4",
}
VERSION_LITERALS = {
    "pokezero.observation.v2",
    "pokezero.observation.v2.1",
    "pokezero.observation.v2.2",
    "pokezero.observation.v3",
    "pokezero.observation.v4",
}
# file::owner -> why naming a version is correct here. A row is a claim that this site wants ONE
# version, not a property, and that a later schema with the same property must NOT match.
ALLOWLIST: dict[str, str] = {
    "src/pokezero/encoding_collision_audit.py::CollisionSketchWriter": (
        "collision-sketch capture REQUIRES v3 and refuses anything else; the point is the exact "
        "version, not a property it happens to have"
    ),
    "src/pokezero/observation.py::<module>": (
        "the schema definitions themselves -- the per-version constants and property tuples are "
        "built here, so naming versions is the definition, not a gate on one"
    ),
    # SCHEMA-KEYED TABLES. Not gates: a table keyed by version is the correct shape. They are
    # listed because adding a schema means REGISTERING IT IN EVERY ONE, and missing one has
    # produced three separate defects in the rotation drill (census maps twice, V2_1_LINEAGE
    # once). The drill's encode-equivalence precondition is what catches a miss empirically;
    # this list is what tells a human where to look.
    "src/pokezero/showdown.py::<module>": (
        "REPLAY_OBSERVATION_SPECS_BY_SCHEMA and the two _MINIMUM_*_CENSUS_BY_SCHEMA maps -- "
        "schema-keyed tables, must be registered when a schema is added"
    ),
    "src/pokezero/mcts_eval/lattice.py::materialize_search_artifacts": (
        "_EXPORTER_SCHEMA_CHOICES -- schema-keyed table; raises ContractError on an unmapped "
        "schema, so an unregistered one fails loudly rather than silently"
    ),
}


def _owner_map(tree: ast.AST) -> dict[int, str]:
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(ln, node.name)
    return owner


def _names_a_version(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in VERSION_NAMES:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in VERSION_NAMES:
            return True
        if isinstance(sub, ast.Constant) and sub.value in VERSION_LITERALS:
            return True
    return False


def scan() -> list[dict]:
    files = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "src/**/*.py", "src/*.py"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    found: list[dict] = []
    for rel in files:
        path = REPO / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            # Loud: an unparsed file is an unscanned file, which is how a guard's denominator rots.
            found.append({"file": rel, "line": 0, "form": "UNPARSED", "owner": type(exc).__name__})
            continue
        owner = _owner_map(tree)
        for node in ast.walk(tree):
            form = None
            if isinstance(node, ast.Compare):
                ops = {type(o).__name__ for o in node.ops}
                # Eq/NotEq/Is/IsNot compare identity; In/NotIn against a LITERAL tuple/list/set
                # is a hand-listed membership, i.e. an identity gate wearing a property's clothes.
                if ops & {"Eq", "NotEq", "Is", "IsNot"} and (
                    _names_a_version(node.left) or any(_names_a_version(c) for c in node.comparators)
                ):
                    form = "compare"
                elif ops & {"In", "NotIn"}:
                    for cmp_node in node.comparators:
                        if isinstance(cmp_node, (ast.Tuple, ast.List, ast.Set)) and _names_a_version(cmp_node):
                            form = "in-literal"
            elif isinstance(node, ast.match_case) and _names_a_version(node.pattern):
                form = "match-case"
            elif isinstance(node, ast.Dict) and any(
                k is not None and _names_a_version(k) for k in node.keys
            ):
                # A dict keyed by versions is a schema-keyed TABLE, which is fine -- but it must be
                # registered when a schema is added, so it is reported for that reason.
                form = "dict-keys"
            if form:
                ln = getattr(node, "lineno", 0)
                key = f"{rel}::{owner.get(ln, '<module>')}"
                if key in ALLOWLIST:
                    continue
                found.append({"file": rel, "line": ln, "form": form, "owner": owner.get(ln, "<module>")})
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = scan()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 1 if rows else 0
    if not rows:
        print("no unlisted schema identity gates in src/")
        return 0
    print(f"{len(rows)} unlisted schema identity gate(s):")
    for r in rows:
        print(f"  {r['file']}:{r['line']}\t{r['form']}\t{r['owner']}")
    print("\nEach asks 'is this schema exactly v<n>?' where it likely means a PROPERTY.")
    print("Convert to membership in a property tuple, or add it to ALLOWLIST with a reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
