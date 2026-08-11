#!/usr/bin/env python3
"""Enumerate every site that reaches the global observation-schema default.

THE DENOMINATOR IS THE POINT. This script exists so no figure about the schema-default
conflation is ever quoted from memory or from a hand-picked file list again. It walks the AST
of every tracked .py file and reports each site with a stable kind, so the count is derived and
re-derivable rather than recalled.

A "site reaching the global default" is any of:

  bare-const     a read of `OBSERVATION_SCHEMA_VERSION` itself
  default-spec   a read of `DEFAULT_REPLAY_OBSERVATION_SPEC` (defined AS the default's spec)
  implicit-spec  `ObservationSpec(...)` with no `schema_version=` -- silently takes the default
  implicit-cfg   `TransformerPolicyConfig(...)` / `.compact_category(...)` with no
                 `observation_schema_version=` -- silently takes the default

Not counted, deliberately: reads of the per-version names (`..._V2_2`, `..._V4`) and of
`SUPPORTED_...`/`REPLAY_OBSERVATION_SPECS_BY_SCHEMA`. Those NAME a schema, which is the state
this migration is moving sites INTO; counting them would make the burndown never converge.

RUN IT UNDER 3.12+, NOT THE VENV. This script only PARSES; it imports nothing from the
package, so any interpreter works -- and `scripts/c153_wide_negative_census.py` uses a
backslash inside an f-string expression, which is legal on 3.12+ and a SyntaxError on 3.11.
Under the 3.11 venv that file lands in UNPARSED and the denominator is incomplete by one.

Usage:  python3.12 scripts/schema_default_ledger.py [--json] [--by-file]
        (any 3.12+ interpreter; the script exits 2 if ANY file failed to parse)
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONST = "OBSERVATION_SCHEMA_VERSION"
DEFAULT_SPEC = "DEFAULT_REPLAY_OBSERVATION_SPEC"
SPEC_CALLS = {"ObservationSpec"}
CFG_CALLS = {"TransformerPolicyConfig", "compact_category"}
# The file that DEFINES the default necessarily reads it; definition sites are not conflation.
DEFINITION_SITES = {"src/pokezero/observation.py"}


def tracked_py() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO / p for p in out]


def enclosing(tree: ast.AST) -> dict[int, str]:
    """line -> enclosing def name, so a ledger row is addressable after line drift."""
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(ln, node.name)
    return owner


def sites_in(path: Path) -> list[dict]:
    rel = str(path.relative_to(REPO))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        # Loud, not skipped: an unparsed file is an unmeasured file, and an unmeasured file is
        # exactly how a denominator goes wrong.
        return [{"file": rel, "line": 0, "kind": "UNPARSED", "owner": type(exc).__name__}]
    owner = enclosing(tree)
    found: list[dict] = []

    def add(node, kind):
        found.append(
            {"file": rel, "line": node.lineno, "kind": kind, "owner": owner.get(node.lineno, "<module>")}
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == CONST and rel not in DEFINITION_SITES:
            add(node, "bare-const")
        elif isinstance(node, ast.Name) and node.id == DEFAULT_SPEC and rel not in DEFINITION_SITES:
            add(node, "default-spec")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            kwargs = {k.arg for k in node.keywords if k.arg}
            if name in SPEC_CALLS and "schema_version" not in kwargs:
                add(node, "implicit-spec")
            elif name in CFG_CALLS and "observation_schema_version" not in kwargs:
                add(node, "implicit-cfg")
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--by-file", action="store_true")
    args = ap.parse_args()

    rows = [r for p in tracked_py() for r in sites_in(p)]
    rows.sort(key=lambda r: (r["file"], r["line"]))

    kinds_all = {r["kind"] for r in rows}
    if args.json:
        print(json.dumps(rows, indent=2))
        # Exit 2 here too. The FIRST version returned 0 from this branch before the UNPARSED
        # check below, so the one output mode the CI gate actually consumes was the one mode
        # without the loud-failure guarantee -- and the gate duly reported the UNPARSED marker
        # row as a brand-new default reader. The discipline has to hold in every mode or it
        # holds in none.
        return 2 if "UNPARSED" in kinds_all else 0

    kinds: dict[str, int] = {}
    files: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        files[r["file"]] = files.get(r["file"], 0) + 1

    if args.by_file:
        for f, n in sorted(files.items(), key=lambda x: (-x[1], x[0])):
            print(f"{n:5d}  {f}")
        print()
    else:
        for r in rows:
            print(f"{r['file']}:{r['line']}\t{r['kind']}\t{r['owner']}")
        print()

    print(f"DENOMINATOR: {len(rows)} sites across {len(files)} files "
          f"(scanned {len(tracked_py())} tracked .py files)")
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {k}")
    if kinds.get("UNPARSED"):
        print("\nWARNING: unparsed files present -- the denominator is INCOMPLETE.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
