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
any verdict. THREE classifications, and the distinction is the point:

  REGISTER  the container means "every supported schema", so the synthetic schema belongs in it.
  PARTIAL   a deliberately chosen subset. `vocab_shift_probe` compares one named PAIR of schemas;
            adding a third does not widen that comparison, it changes what the script does.
  MIRRORED  the drill reaches it through observation.py's property tuples rather than by editing
            the container, so the injection leaves it alone and the mirror carries it.

An earlier form of this docstring said "Two classifications" and listed the first two, while
`classification()` had accepted MIRRORED for eight of the nineteen rows -- a third of the file
described by the enumerator as impossible.

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

# THE CLASS IS THE FIELD THE CENSUS COULD NOT SEE. Matching is on (file, members, variable name);
# the CLASS was read out of the spec and never checked against anything, so flipping a row from
# REGISTER to PARTIAL was exit 0 -- measured, for one row and for all ten -- while the drill's
# registration list silently shrank, because the drill appends only `REGISTER` rows to
# `_reg_targets` and its only guard is `if not _reg_targets` (all ten flipped, not nine). The spec
# is simultaneously this gate's input and its escape hatch: editing a row is how one silences this
# very abort, so it is the likeliest careless change.
#
# NO ORACLE INDEPENDENT OF THE SPEC EXISTS IN THE TREE. Two candidates were built and both are
# REFUTED by the artifact, which is why this is a count pin and not a derivation:
#
#   A. "REGISTER means every supported schema, so its members == SUPPORTED_OBSERVATION_SCHEMA_
#      VERSIONS." False for NINE of the ten REGISTER rows. REGISTER means "a NEW schema belongs
#      here", not "all five are here now": `_EXPORTABLE_TABLE_SCHEMAS` is {v2.2,v3,v4} of five
#      because v2/v2.1 have no exporter tables, and it is still REGISTER.
#   B. "spec MIRRORED == the drill's own `_MIRRORED` tuple." The sets disagree in BOTH directions:
#      5 in the drill, 8 in the spec, with FEATURE_PACK_* and V3_PROJECTION_* in the drill only
#      and SUPPORTED_*, REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA and showdown.py's three maps in
#      the spec only -- they are mirrored by different mechanisms, not by that tuple.
#
# The class encodes an INTENT ("should a schema added tomorrow extend this container?") and no
# artifact in the tree encodes that intent independently. The only true oracle is the runtime
# consequence -- the full drill, 17-70 minutes, which is what found the original nine the
# expensive way. So this pin does not verify the class is RIGHT. It makes a class change LOUD:
# silencing an abort by editing the spec now requires editing this file too, in a differently
# shaped place, with an explicit number a reviewer can question.
#
# PINNED PER FILE, not as three global totals. Three globals were the first form of this pin, and
# review showed the residual they leave is not the single pair the comment disclosed: ANY
# permutation of classes across rows is free. REGISTER<->MIRRORED is the one that matters most --
# those are the two large classes and theirs is the subtle judgement ("does the mirror carry it, or
# must the injection edit it?"), so it is the likeliest accidental reclassification -- and a
# three-cycle (R->M, M->P, P->R) was free too. Both were measured exit 0 against the global form.
#
# Keying the counts by FILE removes every permutation that crosses a file, because a swap between
# two files changes two entries. What remains is a swap WITHIN one file that holds more than one
# class, and today exactly one file does: showdown.py (3 MIRRORED + 1 REGISTER). So the residual is
# now one intra-file swap in one file rather than any pairing in the spec -- stated, because a
# count is exactly as strong as its arithmetic and the last statement of this residual was wrong by
# understatement.
#
# A legitimate reclassification, or a newly classified container, must update this table in the
# same commit. That is the same deliberate trade the spec header records for line-qualified rows:
# a loud reclassify beats a silent free pass.
EXPECTED_CLASS_COUNTS_BY_FILE = {
    "scripts/export_encoder_tables.py": {"REGISTER": 2},
    "scripts/schema_identity_gate_scan.py": {"REGISTER": 1},
    "scripts/vocab_shift_probe.py": {"PARTIAL": 1},
    "src/pokezero/engine_env.py": {"REGISTER": 1},
    "src/pokezero/mcts_eval/lattice.py": {"REGISTER": 1},
    "src/pokezero/mcts_eval/resolver.py": {"REGISTER": 1},
    "src/pokezero/neural_cli.py": {"REGISTER": 2},
    "src/pokezero/observation.py": {"MIRRORED": 5},
    "src/pokezero/rollout_cli.py": {"REGISTER": 1},
    "src/pokezero/showdown.py": {"MIRRORED": 3, "REGISTER": 1},
}


def _assigned_names(tree: ast.AST) -> dict[int, str]:
    """Direct assignment names; nested inline containers use their lexical owner.

    Unnamed containers must never share a classification for free. The former
    line keys distinguished them but drifted when unrelated code moved. Scoped
    keys now include a containing assignment or call when one names the owner;
    indistinguishable duplicates abort, even when placed on separate lines.
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
        if not name:
            continue
        val = node.value
        # `X = frozenset({...})` / `tuple([...])`: the container is the call's first argument.
        if isinstance(val, ast.Call) and val.args:
            val = val.args[0]
        if isinstance(val, (ast.Set, ast.List, ast.Tuple, ast.Dict)):
            out[id(val)] = name
    return out


def _inline_scope(node: ast.AST, parents: dict[int, ast.AST]) -> str:
    """Stable lexical owner, with assignment/call identity when a function has several containers."""
    scopes, owner = [], None
    current = node
    while id(current) in parents:
        current = parents[id(current)]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scopes.append(current.name)
        elif owner is None and isinstance(current, (ast.Assign, ast.AnnAssign)):
            targets = current.targets if isinstance(current, ast.Assign) else [current.target]
            owner = next((t.id for t in targets if isinstance(t, ast.Name)), None)
        elif owner is None and isinstance(current, ast.Call):
            owner = ast.unparse(current.func)
            if current.args and isinstance(current.args[0], ast.Constant) and isinstance(current.args[0].value, str):
                owner += '[' + current.args[0].value + ']'
    scope = '.'.join(reversed(scopes)) or 'module'
    return '<inline:' + scope + ('/' + owner if owner else '') + '>'


def containers() -> tuple[list[tuple[str, int, tuple[str, ...], str]], int]:
    """(file, current line, sorted members, stable name), plus matched AST node count.

    Lines are presentation metadata. Duplicate classification identities abort
    independently of line placement, preserving the anti-free-rider check.
    """
    found: set[tuple[str, int, tuple[str, ...], str]] = set()
    matched_nodes = 0
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
            parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
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
                var = names.get(id(node)) or _inline_scope(node, parents)
                if len(vals) >= 2 and (vals <= SHORT or vals <= FULL):
                    found.add((rel, node.lineno, tuple(sorted(vals)), var))
                    matched_nodes += 1
                elif len(consts) >= 2 and not vals:
                    found.add((rel, node.lineno, tuple(sorted(consts)), var))
                    matched_nodes += 1
    return sorted(found), matched_nodes


def classification() -> dict[tuple[str, tuple[str, ...], str], str]:
    """Committed classification keyed on (file, members, stable named/scoped owner).

    A new owner requires a new row even if its members match an existing owner.
    Unrelated source movement never changes that identity.
    """
    if not SPEC.is_file():
        raise SystemExit(f"census: no classification file at {SPEC}")
    out: dict[tuple[str, tuple[str, ...], str], str] = {}
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
        key = (path, tuple(sorted(members.split(","))), var)
        # A SECOND row for a key already seen used to overwrite the first silently, and the
        # damage was not merely a lost row: `len(spec)` counts dict ENTRIES, so a duplicate made
        # the denominator SMALLER, which is the one direction the count check below cannot see.
        # Measured at 9549cc80: 20 rows of which 2 conflicted -> 19 entries against 19
        # containers -> "every schema-name container is classified", exit 0, with the class
        # decided by whichever row happened to come last in the file. Rows are the record of a
        # human judgement; two contradictory records of one judgement are not a tie for file
        # order to break.
        if key in out:
            if out[key] == kind:
                raise SystemExit(
                    f"census: duplicate row for {path} {var} {list(key[1])} (both {kind}). "
                    "Remove one; a repeated row shrinks the denominator the count check reads."
                )
            raise SystemExit(
                f"census: CONFLICTING rows for {path} {var} {list(key[1])}: "
                f"{out[key]} and {kind}. One container cannot be both; pick one deliberately."
            )
        out[key] = kind
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print every container and its class")
    # THERE IS DELIBERATELY NO `--check`. There was one, and it was a complete no-op: `args.check`
    # was read nowhere, output was byte-identical with and without it (diffed, not reasoned about),
    # and its `--help` text told the reader it "widens the LISTING to every container, not just the
    # unclassified ones" -- which is what `--list` does. A flag whose help claims a behaviour it
    # does not have is the same defect this whole file exists to catch, moved up into the user
    # interface, so it is REMOVED rather than documented as vestigial: a no-op accepted for
    # compatibility still has to be explained to every future reader, and the explanation is the
    # cost. Every abort below applies unconditionally; `--list` only decides how loud the listing
    # is on the way to the same verdict, which is why there is nothing for a second flag to mean.
    args = ap.parse_args()

    found, matched_nodes = containers()
    spec = classification()
    print(f"schema-name literal containers in src/ and scripts/: {len(found)}")
    print(f"classified in {SPEC.relative_to(REPO)}:                {len(spec)}")
    print()

    # A scoped identity cannot silently classify two nodes, even on different lines.
    keys = {(path, members, var) for path, _, members, var in found}
    if matched_nodes != len(found) or len(keys) != len(found):
        print(f"ABORT: {matched_nodes} container node(s) collapsed into {len(keys)} distinct identities. "
              "Give each container a distinct lexical owner; moving lines cannot distinguish them.")
        return 15

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

    # G6: the census printed "19 containers / 18 classified" and then "every container is classified",
    # exit 0 -- a contradiction it stated and ignored. Every container must have its OWN row, so the
    # counts must be equal; a shared row is a free-rider, which is how G3/G4 hid.
    if len(found) != len(spec):
        print()
        print(f"ABORT: {len(found)} container(s) but {len(spec)} classification row(s). Every container "
              "needs its own row -- a shared row means one is classified by another's decision, which "
              "is how a new table gets registered for free.")
        return 15

    # EVERY abort now applies whatever the flags, and `--list` only decides how loud the listing
    # is. A `--check` flag used to gate the three aborts below, so `--list` -- the thing a human
    # reaches for when diagnosing -- printed the stale rows and the unclassified containers of the
    # exact e496e9b8 shape and then exited 0. A diagnostic that shows a contradiction and reports
    # success is the same defect as a gate that states a contradiction and ignores it, which is
    # what G6 was. That flag is gone entirely; see the argparse block above for why.
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

    # The class pin. See EXPECTED_CLASS_COUNTS_BY_FILE: every check above verifies that a row
    # EXISTS for each container, and none of them looks at what the row SAYS.
    actual: dict[str, dict[str, int]] = {}
    for (path, _members, _var), kind in spec.items():
        actual.setdefault(path, {})[kind] = actual.setdefault(path, {}).get(kind, 0) + 1
    if actual != EXPECTED_CLASS_COUNTS_BY_FILE:
        print()
        print(f"ABORT: the per-file classification counts do not match the pin in "
              f"{pathlib.Path(__file__).name}.")
        for path in sorted(set(actual) | set(EXPECTED_CLASS_COUNTS_BY_FILE)):
            want = EXPECTED_CLASS_COUNTS_BY_FILE.get(path)
            got = actual.get(path)
            if want != got:
                print(f"    {path}")
                print(f"      pinned: {want}")
                print(f"      found:  {got}")
        print("The drill registers its synthetic schema in REGISTER rows only, so a REGISTER that "
              "became PARTIAL or MIRRORED shrinks the registration list with no other symptom "
              "until a 17-70 minute drill run charges the omission to the codebase. If the "
              "reclassification is deliberate, change the pin in the same commit and say why.")
        return 15

    totals: dict[str, int] = {}
    for per_file in actual.values():
        for kind, n in per_file.items():
            totals[kind] = totals.get(kind, 0) + n
    print(f"  every schema-name container is classified, and the per-file class counts match the "
          f"pin ({', '.join(f'{k} {totals[k]}' for k in sorted(totals))} "
          f"across {len(actual)} file(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
