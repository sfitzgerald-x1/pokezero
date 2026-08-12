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

Usage:  .venv/bin/python scripts/schema_default_ledger.py [--json] [--by-file]

The script imports nothing from the package -- it only parses -- so any interpreter works. It
exits 2 in EVERY mode if any tracked file fails to parse or is missing, because an unmeasured
file silently shrinks the denominator. (Until #1239 one tracked file could not be parsed on 3.11
and this docstring told readers to avoid the venv; that is no longer true and the advice is
removed rather than left to mislead.)
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
# DERIVED, not hardcoded. The first version listed three call names by hand and therefore
# undercounted by 186 sites -- `LocalShowdownConfig.observation_spec` alone has ~132 callers.
# That is this program's own error class committed inside the instrument built to retire it:
# a denominator chosen rather than enumerated. `derive_surfaces` scans src/ for every class
# attribute or parameter whose DEFAULT is one of GLOBALS, so a new surface is counted the day
# it is written.
GLOBALS = {CONST, DEFAULT_SPEC}
# Alternate constructors do not re-declare the field, so they cannot be derived; they are
# listed against the type they build and asserted to exist.
EXTRA_CONSTRUCTORS = {"TransformerPolicyConfig": ["compact_category"]}


def derive_surfaces() -> dict[str, set[str]]:
    """{callable name -> kwargs that silently default to the global default}."""
    found: dict[str, set[str]] = {}

    def dflt_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id
        return None

    for path in (REPO / "src").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for st in node.body:
                    if isinstance(st, ast.AnnAssign) and st.value is not None:
                        if dflt_name(st.value) in GLOBALS and isinstance(st.target, ast.Name):
                            found.setdefault(node.name, set()).add(st.target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                pairs = list(zip(a.args[-len(a.defaults):] if a.defaults else [], a.defaults))
                pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
                for arg, d in pairs:
                    if dflt_name(d) in GLOBALS:
                        found.setdefault(node.name, set()).add(arg.arg)
    for owner, aliases in EXTRA_CONSTRUCTORS.items():
        if owner not in found:
            raise SystemExit(
                f"ledger: {owner} no longer defaults to the global default; its EXTRA_CONSTRUCTORS "
                "entry is stale and the count would silently drift."
            )
        for alias in aliases:
            found[alias] = found[owner]
    return found


SURFACES = derive_surfaces()
# The file that DEFINES the default necessarily reads it; definition sites are not conflation.
DEFINITION_SITES = {"src/pokezero/observation.py"}


def tracked_py() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO / p for p in out]


def enclosing(tree: ast.AST) -> dict[int, str]:
    """line -> INNERMOST enclosing def/class name, so a row is addressable after line drift.

    Innermost, not outermost. The first version used `ast.walk` (breadth-first) with
    `setdefault`, which locked in the OUTERMOST scope and contradicted this docstring: every
    method of a TestCase collapsed onto the class name, so
    `test_neural_policy.py::NeuralPolicyScaffoldTest::implicit-cfg` covered 54 separate sites as
    one key. 202 rows collapsed to 87 distinct keys -- 115 rows, 57%, invisible to any
    key-based comparison. Assigning unconditionally in depth order makes the innermost scope win.
    """
    owner: dict[int, str] = {}

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                # Unconditional assignment is enough: pre-order visits the shallowest scope
                # first, so a deeper one always overwrites it. A first cut carried a module-level
                # `_DEPTH` dict keyed on `(id(tree), lineno)` to compare depths -- dead weight
                # (the short-circuit meant the depth was never actually read), ~10 MB leaked per
                # scan, and keyed on the `id()` of trees that get freed and reused: 23 id-reuse
                # events across 524 files. Removing the guard clause it hid behind mis-owned 8
                # files, which is how it looked load-bearing.
                owner[ln] = node.name
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return owner


def sites_in(path: Path) -> list[dict]:
    rel = str(path.relative_to(REPO))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError, IsADirectoryError) as exc:
        # Loud, not skipped: an unparsed file is an unmeasured file, and an unmeasured file is
        # exactly how a denominator goes wrong.
        return [{"file": rel, "line": 0, "kind": "UNPARSED", "owner": type(exc).__name__}]
    owner = enclosing(tree)
    found: list[dict] = []

    def add(node, kind, unclosed=None):
        row = {"file": rel, "line": node.lineno, "kind": kind,
               "owner": owner.get(node.lineno, "<module>")}
        if unclosed:
            # Which default-bearing kwarg is still unnamed. Without this a row says "this call
            # reaches a default" without saying through which of several routes.
            row["unclosed"] = unclosed
        found.append(row)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == CONST and rel not in DEFINITION_SITES:
            add(node, "bare-const")
        elif isinstance(node, ast.Name) and node.id == DEFAULT_SPEC and rel not in DEFINITION_SITES:
            add(node, "default-spec")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            kwargs = {k.arg for k in node.keywords if k.arg}
            # EACH-OF, not any-of. `SURFACES[name] & kwargs` scored a call safe if it passed ANY
            # of the surface's default-bearing kwargs, so a `compact_category(numeric_feature_
            # count=..., ...)` that never named a schema counted as migrated while still taking
            # the process-wide default. 43 sites were hidden that way -- and 41 of them pin a
            # WIDTH and default the SCHEMA, which is precisely the shape of the two production
            # bugs (#1227 token_count, #1228 the feature widths) this ledger cites as its reason
            # to exist. A site is only safe once EVERY route to a global is closed.
            unclosed = SURFACES.get(name, frozenset()) - kwargs
            if name in SURFACES and unclosed:
                # One kind per surface so a new surface cannot quietly join an existing bucket.
                add(node, f"implicit:{name}", sorted(unclosed))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--by-file", action="store_true")
    args = ap.parse_args()

    scanned = tracked_py()
    rows = [r for p in scanned for r in sites_in(p)]
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
          f"(scanned {len(scanned)} tracked .py files)")
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {k}")
    if kinds.get("UNPARSED"):
        print("\nWARNING: unparsed files present -- the denominator is INCOMPLETE.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
