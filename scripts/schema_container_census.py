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

# THE CLASS IS THE FIELD `--check` COULD NOT SEE. Matching is on (file, members, variable name);
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
    """Map a container node's id() to the variable it is assigned to, where there is one.

    The classification is keyed on the container's NAME as well as its members. Keying on members
    alone let a NEW routing tuple whose member set duplicated an existing row be classified for free
    and then checked by nothing -- which is drill defect #1 (an unregistered routing tuple) reopening
    through the very file added to prevent it. A container with no assignment target (an inline set in
    an `if x not in {...}`) gets "<inline:LINE>" -- see the caller at the `var =` line below, which
    supplies the line number this function cannot see.

    An earlier form of this docstring said such a container gets a bare "<inline>" and that "two
    inline containers in the same file with the same members are genuinely the same classification
    question". BOTH halves were false, and in the direction that invites a maintainer to delete the
    line qualification: the bare key collided neural_cli.py's train tuple with its iterate tuple and
    let a third container be classified for free, which is why the line was added. Two unnamed
    containers with the same members in the same file are NOT the same question -- they are the
    free-rider case this key exists to separate.

    KNOWN GAP, since this docstring is the place a reader looks for the key's guarantees: distinctness
    is still per (file, lineno, members, name), so two unnamed containers on the SAME physical line
    with the SAME members KEY IDENTICALLY and one spec row covers both. That much is unchanged.

    WHAT CHANGED is the consequence, and this paragraph used to overstate the damage in the reader's
    favour: it said the collapse was "exit 0 while printing 20 containers / 20 rows on a tree holding
    21". That was measured and true when written, and it is now FALSE -- `main()` compares the number
    of matched AST nodes against the number of distinct entries and aborts 15 on the difference. The
    census can no longer report a denominator it knows is short. It still cannot tell the two
    containers APART, so the abort says "put them on separate lines" rather than classifying both.

    A stable per-(file, members) ordinal -- `<inline#1>`, `<inline#2>` -- remains the candidate fix
    for telling them apart, and it remains a PREDICTION, not a measured result: nobody has built it.
    What IS measured is the negative that motivates it -- simply dropping the line qualification is
    exit 0 for a newly added unregistered container, so line-insensitivity alone is not the answer.
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


def containers() -> tuple[list[tuple[str, int, tuple[str, ...], str]], int]:
    """(file, line, sorted members, variable name) for every literal container of schema names.

    Returns the deduped entries AND the number of AST nodes that produced them. The two differ
    only in the collapse case documented on `_assigned_names` -- two unnamed containers on the
    SAME physical line with the SAME members key identically -- and the caller aborts on the
    difference. Detecting the collapse is NOT the per-(file, members) ordinal that docstring
    predicts; it does not tell the two apart, it refuses to report a denominator it knows is
    short. That is the whole of the fix: the measured symptom was exit 0 while printing "20
    containers / 20 rows" on a tree holding 21, and a count the tool cannot stand behind must
    abort rather than be printed.
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
                # LINE-QUALIFIED when there is no assignment target. A bare "<inline>" collided two genuinely
                # different argparse tuples in neural_cli.py (train at :467, iterate at :1952) into ONE
                # classification row -- so a third was classified for free, and the injection, which runs
                # once per row, inserted the synthetic schema TWICE into each. Also resolve through call
                # wrappers: `frozenset({...})` / `tuple([...])` put the container inside a Call, so
                # `_EXPORTABLE_TABLE_SCHEMAS` -- one of the nine historical misses -- was "<inline>" too.
                var = names.get(id(node)) or f"<inline:{node.lineno}>"
                if len(vals) >= 2 and (vals <= SHORT or vals <= FULL):
                    found.add((rel, node.lineno, tuple(sorted(vals)), var))
                    matched_nodes += 1
                elif len(consts) >= 2 and not vals:
                    found.add((rel, node.lineno, tuple(sorted(consts)), var))
                    matched_nodes += 1
    return sorted(found), matched_nodes


def classification() -> dict[tuple[str, tuple[str, ...], str], str]:
    """Committed classification, keyed on (file, members, variable name).

    The variable name is in the key on purpose; keying on (file, members) alone let a NEW routing
    tuple whose member set duplicated an existing row be classified for free. For a container with
    no assignment target that name is `<inline:LINE>`, so those rows -- and ONLY those -- are line
    sensitive, which the spec file's header states as a deliberate trade: "a drifting line is a loud
    reclassify, whereas a silent collision is a free pass." An earlier form of this docstring said
    the key was "NOT on line, which drifts", which described neither the key nor the intent and
    invited reading the resulting abort as a census bug rather than as the reclassify prompt it is.
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
    ap.add_argument(
        "--check", action="store_true",
        help="kept for the drill's invocation and for readability at the call site; every abort "
             "applies with or without it. It widens the LISTING to every container, not just the "
             "unclassified ones",
    )
    args = ap.parse_args()

    found, matched_nodes = containers()
    spec = classification()
    print(f"schema-name literal containers in src/ and scripts/: {len(found)}")
    print(f"classified in {SPEC.relative_to(REPO)}:                {len(spec)}")
    print()

    # The denominator must be one the tool can stand behind. `matched_nodes` counts the AST nodes
    # that matched; `found` is those nodes deduped by (file, line, members, name). A difference is
    # the collapse case: two unnamed containers on the SAME physical line with the SAME members.
    # Measured before this check existed: a file adding two such containers made the census print
    # "20 containers / 20 rows" and exit 0 on a tree holding 21, one spec row covering both -- the
    # free-rider shape the key was made finer to prevent, surviving at same-line granularity.
    if matched_nodes != len(found):
        print(f"ABORT: {matched_nodes} container node(s) collapsed into {len(found)} distinct "
              "entr(ies). Two unnamed containers on the SAME line with the SAME members key "
              "identically, so one spec row would cover both and the printed count is short. "
              "Put them on separate lines so each gets its own <inline:LINE> key. An earlier form "
              "of this message also offered 'or give at least one an assignment target', which is "
              "not always possible -- a literal nested inside a list has nowhere to put one.")
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

    # G6: `--check` printed "19 containers / 18 classified" and then "every container is classified",
    # exit 0 -- a contradiction it stated and ignored. Every container must have its OWN row, so the
    # counts must be equal; a shared row is a free-rider, which is how G3/G4 hid.
    if len(found) != len(spec):
        print()
        print(f"ABORT: {len(found)} container(s) but {len(spec)} classification row(s). Every container "
              "needs its own row -- a shared row means one is classified by another's decision, which "
              "is how a new table gets registered for free.")
        return 15

    # EVERY abort now applies whatever the flags, and `--check` only decides how loud the listing
    # is. It used to gate the three aborts below, so `--list` -- the thing a human reaches for when
    # diagnosing -- printed the stale rows and the unclassified containers of the exact e496e9b8
    # shape and then exited 0. A diagnostic that shows a contradiction and reports success is the
    # same defect as a gate that states a contradiction and ignores it, which is what G6 was.
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
