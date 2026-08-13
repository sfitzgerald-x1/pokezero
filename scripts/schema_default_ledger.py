#!/usr/bin/env python3
"""Enumerate every site that reaches the global observation-schema default.

THE DENOMINATOR IS THE POINT. This script exists so no figure about the schema-default
conflation is ever quoted from memory or from a hand-picked file list again. It walks the AST
of every tracked .py file and reports each site with a stable kind, so the count is derived and
re-derivable rather than recalled.

A "site reaching the global default" is any of:

  bare-const           a read of `OBSERVATION_SCHEMA_VERSION` itself                    (16)
  default-spec         a read of `DEFAULT_REPLAY_OBSERVATION_SPEC`                        (54)
  implicit:<Surface>   a call to <Surface> leaving at least one default-bearing kwarg unnamed,
                       one kind per surface so a new one cannot join an existing bucket. All EIGHT
                       surfaces, with provenance -- seven DERIVED from src/, one an alternate
                       constructor that cannot be (marked *):

                         LocalShowdownConfig            133    ObservationSpec            34
                         compact_category *              96    TransformerPolicyConfig     4
                         PokeZeroObservationV0           49    LinearPolicyModel           3
                                                               OnlineBattleAgent           2
                         observation_from_player_state    0

                       `observation_from_player_state` is DERIVED and carries ZERO rows: all 130 of
                       its call sites pass `spec=` explicitly. A modelled surface with no open sites
                       is the goal state, not an absence -- it is listed so that a caller which
                       later drops `spec=` lands in a named bucket instead of appearing to be a new
                       surface. An earlier version of this table said "all seven derived surfaces"
                       while listing compact_category (not derived) and omitting this one: the count
                       was right and the membership was not.

                       The row's `unclosed` field names which kwarg is still open. These nine counts
                       are pinned by tests/test_schema_default_ledger.py -- a previous docstring
                       table staled for several commits because nothing held it to the tool.

  Two retired names, `implicit-spec` and `implicit-cfg`, were listed here long after the code
  stopped emitting them -- the reader-facing vocabulary staled in the same change that derived the
  surfaces and recovered 187 sites. Worse, a commit message claimed this list had been updated when
  the edit had silently no-op'd: a `str.replace` whose target no longer existed, with no assertion
  that it applied. Every figure above is re-derivable by running this script.

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
# undercounted by 187 sites -- `LocalShowdownConfig.observation_spec` alone has ~133 callers.
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
        if isinstance(node, ast.Attribute):
            # BOTH sides. `DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count` puts the
            # global on the VALUE side (2 sites: neural_policy.py:245-246);
            # `observation.OBSERVATION_SCHEMA_VERSION` puts it on the ATTR side.
            #
            # The ATTR-side arm currently matches ZERO sites. It was added with a comment claiming
            # the spelling was "an idiom used 31 times in this repo", which is false in two ways:
            # no default anywhere in src/ is written module-qualified, and no dotted read of either
            # global exists in the tree at all --
            #   git grep -hoE '\w+\.(OBSERVATION_SCHEMA_VERSION|DEFAULT_REPLAY_OBSERVATION_SPEC)\b' \
            #     -- '*.py' | wc -l   ->   0
            # The 31 appears to have been a garbled recollection of the aliased module import,
            # which is 24 (see docs/schema_default_ledger.md). The arm is KEPT -- a surface with
            # a module-qualified default would otherwise be derived-blind, and this file's whole
            # thesis is that unmeasured spellings are how the denominator goes wrong -- but it is a
            # guard against a spelling that does not occur yet, not a fix for observed sites. It is
            # pinned by test_schema_default_ledger.py, not by this prose.
            if node.attr in GLOBALS:
                return node.attr
            if isinstance(node.value, ast.Name):
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
    one TestCase's single key covered 54 separate call sites as
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


def base_root(node: ast.AST) -> str | None:
    """Leftmost Name of a pure Name/Attribute chain: `a.b.c` -> "a", `f().b` -> None.

    Walking to the ROOT rather than testing `isinstance(node.value, ast.Name)` is what makes an
    arbitrarily deep base match. The one-level test could only see `O.GLOBAL`, so every dotted
    base -- `pokezero.observation.GLOBAL`, `pokezero.showdown.GLOBAL.numeric_feature_count` --
    read the default with the denominator unmoved and the authorship gate green.
    """
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


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

    # Per-file ALIAS map. `from pokezero.observation import OBSERVATION_SCHEMA_VERSION as SV`
    # made every `SV` read invisible, and `import pokezero.observation as O` made every
    # `O.OBSERVATION_SCHEMA_VERSION` invisible. Both were demonstrated to add default reads with
    # N unchanged and the gate green -- the same defect class as the any-of bug: a denominator
    # blind to a spelling. Resolved here so the kind reported is the GLOBAL, not the local name.
    alias_to_global: dict[str, str] = {}
    # Names bound in this file that a pokezero module (or the package) can be reached THROUGH.
    # Matched by ROOT segment, not by whole dotted path: `import pokezero` binds only `pokezero`,
    # yet `pokezero.observation.OBSERVATION_SCHEMA_VERSION` is then a legal read, so requiring the
    # full base to have been imported by name is itself a spelling hole.
    module_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in (CONST, DEFAULT_SPEC) and a.asname:
                    alias_to_global[a.asname] = a.name
                elif node.module and node.module.startswith("pokezero"):
                    # `from pokezero import observation [as O]`. Not restricted to a hand-listed
                    # pair of module names: that list was ("observation", "showdown"), so
                    # `from pokezero import replay` was blind, and the whole point of this map is
                    # that the set of spellings is not something to enumerate from memory.
                    module_roots.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                # `import pokezero` -- no dot -- binds the package and re-exports both globals
                # via __all__. The previous `startswith("pokezero.")` test registered NOTHING for
                # it, making the public spelling the one the ledger could not see.
                if a.name == "pokezero" or a.name.startswith("pokezero."):
                    module_roots.add(a.asname or a.name.split(".")[0])

    def add(node, kind, unclosed=None):
        row = {"file": rel, "line": node.lineno, "kind": kind,
               "owner": owner.get(node.lineno, "<module>")}
        if unclosed:
            # Which default-bearing kwarg is still unnamed. Without this a row says "this call
            # reaches a default" without saying through which of several routes.
            row["unclosed"] = unclosed
        found.append(row)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and rel not in DEFINITION_SITES and (
            node.id == CONST or alias_to_global.get(node.id) == CONST
        ):
            add(node, "bare-const")
        elif isinstance(node, ast.Name) and rel not in DEFINITION_SITES and (
            node.id == DEFAULT_SPEC or alias_to_global.get(node.id) == DEFAULT_SPEC
        ):
            add(node, "default-spec")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in (CONST, DEFAULT_SPEC)
            and base_root(node.value) in module_roots
            and rel not in DEFINITION_SITES
        ):
            add(node, "bare-const" if node.attr == CONST else "default-spec")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            kwargs = {k.arg for k in node.keywords if k.arg}
            # EACH-OF, not any-of. `SURFACES[name] & kwargs` scored a call safe if it passed ANY
            # of the surface's default-bearing kwargs, so a `compact_category(numeric_feature_
            # count=..., ...)` that never named a schema counted as migrated while still taking
            # the process-wide default. TWO figures, which are easy to conflate and were:
            #   43 sites were hidden by the any-of bug; 41 of those 43 pin a WIDTH and default the
            #      SCHEMA -- precisely the shape of #1227 (token_count) and #1228 (the widths).
            #   127 of ALL 391 rows have only a SCHEMA route open, under either kwarg name
            #      (41 compact_category + 49 PokeZeroObservationV0 + 34 ObservationSpec +
            #      3 LinearPolicyModel). Of those, 44 are open specifically on
            #      `observation_schema_version`. An earlier comment quoted the 44 while
            #      describing the 127's question -- a true number answering a narrower question
            #      than the one asked, which is the failure this whole ledger exists to retire.
            # A site is only safe once EVERY route to a global is closed.
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
