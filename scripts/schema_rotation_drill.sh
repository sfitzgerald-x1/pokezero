#!/usr/bin/env bash
# Phase D acceptance drill: prove the schema-default conflation is dead.
#
# Rotate the default to a schema NOTHING names, run the suite, and count what breaks. Only the
# class-(iii) legitimate readers -- the sites whose job is to answer "nobody said" -- may notice.
# Anything else that breaks is a site that silently depended on which schema held the default
# slot, i.e. a surviving instance of the class, and it earns a ledger row.
#
# A synthetic v5 is used rather than v4 on purpose: v4 is named all over the tree, so rotating
# to it cannot distinguish "names v4 deliberately" from "happens to match the default". A schema
# nobody names is the only clean probe, and it is also the real future event this must make free.
#
# Usage:  bash scripts/schema_rotation_drill.sh [worktree-path]
#
# NO --ignore list. It previously skipped the three modules that could not be COLLECTED under
# the 3.11 venv, so the drill scored a suite with three files silently absent -- the same
# narrowed denominator this whole program exists to retire, inside its own acceptance check.
# Those modules are collectable again (the c153 f-string fix), and if any file ever becomes
# uncollectable the run now fails on the ERROR guard instead of quietly shrinking.
#         DRILL_SCOPE=fast   ...  scope to the files that have ever broken + the expected set.
#         DRILL_SHAPE=differ ...  VERIFIED; see STATUS below and the verdict report.
#
# Why a shape-differing variant exists. The default synthetic schema is shape-IDENTICAL to v4 so
# that every breakage is unambiguously a NAMING failure. The cost of that choice is that no
# shape-dependent conflation can break under it -- and both real bugs this effort found
# (#1227 token_count, #1228 the feature widths) were shape bugs. A drill that structurally
# cannot reproduce its own motivating defects is not sufficient evidence on its own.
#
# STATUS: `differ` is CONSTRUCTED AND VERIFIED. It has been run to completion, three arms, and its
#         rubric is populated per its own admission rule -- see
#         reports/schema_rotation_drill_verdict.md. Its coverage is ONE site wide, which is a fact
#         about how little of this suite pins the default's width, not a claim about the arm.
# a real run on a tree where the drill can execute, and the exit-12 INCONCLUSIVE gate below
# enforces that mechanically -- it cannot print a PASS-shaped summary.
#
# The distinction matters and the previous version of this note blurred it. It said differ was "NOT
# sound" and named two open causes for the contradiction observed on its first run (a test appearing
# in BOTH "unexpected breakages" and "expected but did not break"). All THREE causes now known are
# closed, so a reader following that note would re-investigate fixed code:
#
#   the FAILED -> ERROR bucket change on the seven fallback_replay tests
#       -- closed: `_norm_id` strips both prefixes, so the two buckets normalise to ONE id.
#          Verified by feeding it a FAILED and an ERROR line for the same test.
#   the baseline reuse key covering (SHA, scope) but not shape
#       -- closed: the stamp is (SHA, scope, SHAPE, interpreter), written after both runs.
#   the ACTUAL cause, which neither of those was
#       -- closed: `_norm_id` retained pytest's ` - <reason>` suffix, so no id could ever match the
#          rubric's bare ids. Every breakage landed in UNEXPECTED and every rubric row in MISSING,
#          which IS "appearing in both". Found by adversarial review, not by me; I had attributed
#          the symptom to relative-vs-absolute paths and declared it fixed.
#
# The construction is also complete on the census axis, which was the other half of the ask: the
# differ spec narrows `numeric_feature_count` by one AND `_MINIMUM_NUMERIC_CENSUS_BY_SCHEMA` by one,
# so the spec and its floor move together rather than the floor refusing the schema it describes;
# and differ has its own pass condition, because the class-(iii) set does not apply under a width
# change. Its rubric was committed EMPTY for that reason -- populating it from an unverified run
# would launder a guess into a pin.
#
# So what remains is verification, not construction, and it needs a tree where the drill runs.
# The acceptance test is specific: a canary asserting `numeric_feature_count == 132` must break
# under the differ arm (131 columns there) and must not break under the identical arm. Until that
# was observed, the shape half of the class was UNCOVERED. It HAS now been observed: a width pin
# passes under `identical` and fails under `differ` (PRECONDITION 3), and one real site differs
# between the arms. The paragraph above is kept as the reason the rubric started empty.
#
# `fast` exists because two full suites take ~22 minutes and get killed by most runners. It is
# for ITERATION ONLY and is NOT the stop condition: a scoped drill cannot see a NEW breakage in
# a file that has never broken before, which is precisely what the full drill is for. Treat a
# fast PASS as "worth running the full one", never as proof.
# Exit 0 = the class is dead. Nonzero = breakages beyond the legitimate readers; see the diff.
set -uo pipefail

# Run from a COPY. Bash reads a script incrementally from a byte offset, so editing this file
# while a run is in flight shifts the remaining bytes and the running shell dies mid-script with
# a bogus syntax error -- which happened, and invalidated a 35-minute run. Re-execing from an
# immutable snapshot makes the run immune to edits in the working tree.
if [ "${DRILL_REEXEC:-0}" != "1" ]; then
  _snap="$(mktemp -t schema_rotation_drill)" || exit 3
  cat "${BASH_SOURCE[0]}" > "$_snap" || { echo "ABORT: snapshot copy failed"; exit 10; }
  # `bash <empty file>` exits 0, so an unchecked copy turns disk-full into a silent PASS.
  [ -s "$_snap" ] || { echo "ABORT: snapshot is empty"; exit 10; }
  trap 'rm -f "$_snap"' EXIT
  # DRILL_REPO must ride along: $REPO is derived from BASH_SOURCE, which after re-exec points at
  # the snapshot in /tmp, not the checkout. The first cut omitted it and the run died on
  # "fatal: not a git repository".
  DRILL_REEXEC=1 DRILL_REPO="${DRILL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" \
    exec bash "$_snap" "$@"
fi

REPO="${DRILL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WT="${1:-/tmp/schema-v5-drill}"
VENV="$REPO/.venv/bin/python"

# ONE normaliser for every comparison. The previous `sed 's|.*/tests/||'` assumed pytest
# printed ABSOLUTE paths; when it printed relative ones the pattern did not match, so
# `actual.txt` held "FAILED tests/x.py::y" while `expected.txt` held "x.py::y" and NOTHING
# compared equal. The visible symptom was a test appearing in BOTH "unexpected breakages" and
# "expected but did not break" -- impossible, and the tell that the comparison, not the tree,
# was broken. Strips the status word and any leading path through the last `tests/`.
# Delimiter is '#', not '|': the first cut wrote s|^(FAILED|ERROR)...| and sed read the
# alternation's pipe as the closing delimiter -- "RE error: parentheses not balanced", printed
# to stderr while the pipeline carried on and produced EMPTY normalised files. Empty files
# compare as "0 breakages", which is indistinguishable from a pass. A scoring bug that fails
# toward PASS is the worst kind, and this one did.
# Keeps the SUBTEST PARAMETER: `SUBFAILED(row='R1') <id>` normalises to `<id>[row='R1']`, not to
# `<id>`. Stripping it collapsed 24 subtest rows onto one id, so a single failing subtest in the
# baseline excused every OTHER subtest of the same test in the rotated arm -- a real regression
# inside a partially-broken test would have been subtracted silently. The artifact file still
# excuses at test granularity when that is genuinely right, but the drill no longer does it by
# accident.
_norm_id() {
  # The ` - <reason>` SUFFIX is stripped FIRST. pytest -q appends the assertion text to every
  # FAILED/SUBFAILED line, so ids carried `... - AssertionError: 3 != 4` and could never match the
  # rubric's bare ids: every breakage landed in UNEXPECTED and all 7 rubric rows in MISSING.
  #
  # That is exactly the "a test appears in BOTH unexpected breakages and expected-but-did-not-break"
  # symptom this file's header attributes to relative-vs-absolute paths and declares fixed. The path
  # normalisation was real but it was not the cause; this is. It also saturated
  # EXPECTED-BUT-DID-NOT-BREAK, which the rubric header calls the only guard against a pin silently
  # going stale -- so the dead-pin detector could not fire either.
  #
  # Width-dependent, too: pytest truncates the reason to the terminal width (80 in a pipe), so the
  # unnormalised ids differed between a tty and a pipe.
  #
  # The end-of-line anchors are `$`, NOT `\$`. They were written escaped, and inside sed's
  # single-quoted expression `\$` matches a LITERAL DOLLAR SIGN -- so the suffix strip and the whole
  # SUBFAILED rewrite matched nothing, and this function was INERT while the comment above it
  # declared the defect fixed. Measured on real lines from a full run:
  #
  #   in   FAILED /tmp/.../tests/test_observation.py::T::test_x - AssertionError: 3 != 4
  #   was  test_observation.py::T::test_x - AssertionError: 3 != 4      <- reason still attached
  #   now  test_observation.py::T::test_x
  #
  #   in   SUBFAILED(row='R8') ../../../tmp/.../tests/test_unreachable_readjudication.py::T::t
  #   was  SUBFAILED(row='R8') ../../../tmp/.../tests/test_unreachable_readjudication.py::T::t
  #   now  test_unreachable_readjudication.py::T::t[row='R8']
  #
  # Self-tested by --self-test below, on those exact strings, so this cannot go inert again quietly.
  # Structure, not a sequence of nibbles. Every pytest line is
  #     <MARKER>[(<param>)] <SPACE> <test-id> [ - <reason>]
  # and the test id is the FIRST whitespace-delimited token after the marker. So each rule takes the
  # marker, captures the param GREEDILY to the last `)` that is followed by whitespace, keeps the
  # next token, and discards the rest -- which strips the reason as a side effect of keeping only
  # the id, rather than by matching the reason's shape.
  #
  # The previous version stripped ` - <reason>` FIRST and captured the param with `[^)]*`. Both
  # failed on real subtest parameters, which are prose:
  #
  #   SUBFAILED(spelling='UN-annotated class attribute (ast.Assign, not AnnAssign)') <path>::<id>
  #       `[^)]*` stopped at the paren inside "(ast.Assign", so the rewrite produced
  #       "...not AnnAssign)') <path>" -- an id matching nothing, reported as an unexpected breakage.
  #   SUBFAILED(spelling='dataclasses field(default=...) -- 201 `field(default` uses in src/')
  #       the reason strip fired on the " -- 201" INSIDE the parameter and truncated the line.
  #
  # My self-test had five cases and none carried a paren or a dash in the parameter, so it passed
  # while the normaliser mangled four real ids. The cases below are taken verbatim from a full run.
  sed -E '
    s#^SUBFAILED\((.*)\)[[:space:]]+([^[:space:]]+).*$#\2[\1]#
    s#^(FAILED|ERROR)[[:space:]]+([^[:space:]]+).*$#\2#
    s#^SUBFAILED[[:space:]]+([^[:space:]]+).*$#\1#
    s#(^|::)[^ ]*/tests/#\1#
    s#^tests/##
  '
}

# A normaliser whose output feeds BOTH the scorer and the rubric comparison is the single point
# where the whole verdict can go silently wrong: if ids do not match the rubric, every breakage
# reports as UNEXPECTED and every rubric row as MISSING, which reads as "the migration is
# incomplete AND every pin is dead" rather than as a broken sed. It went inert exactly that way.
# Runs before any suite, costs milliseconds.
_norm_self_test() {
  # `n` is COUNTED, not written. The first version printed a hardcoded "5/5" while the case list had
  # grown to nine -- a figure quoted rather than derived, inside the guard whose whole job is to stop
  # exactly that. It would have read "5/5" with four cases silently unexercised.
  local bad=0 n=0 got want
  while IFS='|' read -r raw want; do
    [ -z "$raw" ] && continue
    n=$((n + 1))
    got=$(printf '%s\n' "$raw" | _norm_id)
    if [ "$got" != "$want" ]; then
      echo "  NORMALISER SELF-TEST FAILED"
      echo "    in:   $raw"
      echo "    want: $want"
      echo "    got:  $got"
      bad=1
    fi
  done <<'CASES'
FAILED /tmp/schema-v5-drill/tests/test_observation.py::T::test_x - AssertionError: 3 != 4|test_observation.py::T::test_x
SUBFAILED(row='R8') ../../../tmp/schema-v5-drill/tests/test_unreachable.py::T::t|test_unreachable.py::T::t[row='R8']
FAILED tests/test_observation.py::T::t|test_observation.py::T::t
ERROR /abs/path/tests/test_a.py::T::t|test_a.py::T::t
FAILED tests/test_b.py::T::t_with_dash - E   assert 1 == 2|test_b.py::T::t_with_dash
SUBFAILED(spelling='UN-annotated class attribute (ast.Assign, not AnnAssign)') ../../../tmp/schema-v5-drill/tests/test_led.py::C::t|test_led.py::C::t[spelling='UN-annotated class attribute (ast.Assign, not AnnAssign)']
SUBFAILED(spelling='dataclasses field(default=...) -- 201 `field(default` uses in src/') ../../../tmp/x/tests/test_led.py::C::t|test_led.py::C::t[spelling='dataclasses field(default=...) -- 201 `field(default` uses in src/']
SUBFAILED(spelling='field(default_factory=lambda: ...)') /abs/tests/test_led.py::C::t|test_led.py::C::t[spelling='field(default_factory=lambda: ...)']
SUBFAILED(kind='bare-const') tests/test_led.py::C::t - AssertionError: 17 != 16|test_led.py::C::t[kind='bare-const']
CASES
  [ "$bad" = 0 ] || {
    echo "ABORT: the id normaliser does not do what the scorer assumes. Every comparison against"
    echo "       the rubric would be meaningless, in the direction that reports a false failure."
    exit 14
  }
  echo "  id normaliser self-test: $((n - bad))/$n"
}

drill_targets() {
  local root="$1"
  if [ "${DRILL_SCOPE:-full}" = "fast" ]; then
    sed 's/::.*//' "$REPO/tests/data/schema_drill_scope.txt" 2>/dev/null \
      | grep -vE '^\s*(#|$)' | sort -u | sed "s|^|$root/tests/|"
  else
    echo "$root/tests"
  fi
}

echo "== drill: synthetic v5 rotation =="
_norm_self_test
git -C "$REPO" worktree remove --force "$WT" 2>/dev/null
git -C "$REPO" worktree add -q --detach "$WT" HEAD || exit 3
cd "$WT" || exit 3
export DRILL_WT="$WT"

# Written to a FILE and invoked twice -- once for the treatment arm, once for the control arm --
# rather than duplicated or re-extracted from this script with sed. Re-extraction was the first cut:
# it worked, and it meant the control arm's injection was a 226-line block located by two text
# anchors, so any drift in either would have silently injected a PARTIAL registration and charged
# the resulting breakages to the rotation. The two arms must run byte-identical injection code or
# the subtraction between them means nothing.
INJ="$(mktemp -t drill_inject)" || exit 3
trap 'rm -f "$INJ"' EXIT
cat > "$INJ" <<'PY'
import ast, pathlib, re, sys
wt = sys.argv[1]
p = f"{wt}/src/pokezero/observation.py"
s = open(p).read()

# A v5 that is byte-identical IN SHAPE AND ROUTING TO THE SCHEMA BEING ROTATED AWAY FROM. The
# drill is about NAMING, not about layout: any shape difference makes a breakage ambiguous between
# "reached the default" and "assumed a layout", and identical shape is what makes every breakage
# unambiguously a naming failure.
#
# This used to clone v4 unconditionally, which satisfies the premise ONLY on a tree whose default
# is already v4. On the pre-rotation tree the default is v2.2, so the "identical" arm moved the
# default's shape from 155/51/151 to 132/41/23 -- every width and layout breakage in it was a
# SHAPE artifact being reported as a naming failure, and the drill's central claim was false while
# its own precondition passed, because that precondition also compared against v4 (see
# PRECONDITION 2). Fifth instance of the instrument manufacturing the failure it reports, and the
# only one that reached the verdict.
#
# Everything below therefore mirrors the OUTGOING DEFAULT: its spec, its four routing properties,
# its transition-token count, its census floors. `identical` means identical to what was replaced.
anchor = re.search(r'^OBSERVATION_SCHEMA_VERSION_V4\s*=.*$', s, re.M)
if anchor is None:
    raise SystemExit(
        "drill: could not locate `OBSERVATION_SCHEMA_VERSION_V4 = ...` to anchor the synthetic "
        "schema after. Without the anchor there is no v5-drill constant to rotate to."
    )
s = s[:anchor.end()] + '\nOBSERVATION_SCHEMA_VERSION_V5_DRILL = "pokezero.observation.v5-drill"' + s[anchor.end():]
# Appended LAST, not first. `schema_with()` iterates `reversed(SUPPORTED)` to prefer the NEWEST
# match, so inserting the synthetic schema at the front made it the oldest and `schema_with()`
# never returned it -- every site the migration moved to the property selector was INERT under
# the drill, which is the migration's own vocabulary going untested by its acceptance check.
_m = re.search(r'^SUPPORTED_OBSERVATION_SCHEMA_VERSIONS\s*=\s*\((.*?)\)', s, re.S | re.M)
if _m is None:
    raise SystemExit("drill: could not locate SUPPORTED_OBSERVATION_SCHEMA_VERSIONS")
s = s[:_m.end(1)] + "\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL," + s[_m.end(1):]
# Rotate WHATEVER the current default is, not a hardcoded v4. The previous line replaced the
# literal `= OBSERVATION_SCHEMA_VERSION_V4`, which matches 0 times on any tree whose default is
# not already v4 -- so the default never rotated, and precondition 1 aborted at exit 8. That made
# the drill unrunnable until the rotation itself had landed, which is backwards: the drill exists
# to decide whether the rotation is safe, so it must run BEFORE it, on the tree as it stands.
#
# Anchored on the definition line, so it cannot match a per-version constant or a comment.
_mdef = re.search(r'^OBSERVATION_SCHEMA_VERSION = (OBSERVATION_SCHEMA_VERSION_V\w+)$', s, re.M)
if _mdef is None:
    raise SystemExit(
        "drill: could not locate the default definition line "
        "`OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_<V>`. The injection cannot "
        "rotate a default it cannot find, and rotating nothing would score a PASS against an "
        "unrotated tree."
    )
import os as _os  # noqa: E402 - needed here, before the control-arm branch below
_out = _mdef.group(1)
_suffix = _out[len("OBSERVATION_SCHEMA_VERSION_"):]
_control = _os.environ.get("DRILL_NO_ROTATE") == "1"
if not _control:
    print(f"drill: rotating the default away from {_out}")
# DRILL_NO_ROTATE is the CONTROL ARM: register the synthetic schema exactly as the treatment does,
# and leave the default alone. Anything that breaks in the control breaks because a schema was
# ADDED, not because the default MOVED, and must not be charged to the rotation. Without this the
# drill attributes both to the rotation -- see the control-subtraction block below for what that
# cost.
if _control:
    print(f"drill: CONTROL ARM -- schema registered, default deliberately LEFT at {_out}")
else:
    s = s[:_mdef.start()] + "OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V5_DRILL" + s[_mdef.end():]

# Mirror the OUTGOING DEFAULT's routing properties, computed rather than hardcoded. This replaces
# four separate regex insertions that each added v5-drill wherever v4 appeared: GROUPED_LAYOUT,
# FEATURE_PACK, V2_1_LINEAGE and the transition-token table. Every one of them encoded "the
# outgoing default is v4", so on a v2.2 tree they gave the synthetic schema v4's routing -- which
# is the shape/naming ambiguity this drill exists to exclude.
#
# Appended at END of the module, after every tuple is defined: a mirror has to READ the tuples,
# and referencing one from inside its own literal is a NameError. `V2_1_LINEAGE` is the member that
# taught this -- it was missed by the hardcoded set, drove `schema_v2_1` in showdown.py, and made
# the synthetic schema encode two numeric cells differently from its reference. Deriving membership
# instead of listing it is what stops the next such tuple from being missed: a new property tuple
# is mirrored automatically as long as it is named here, and PRECONDITION 2 fails loudly if it is
# not.
# `transition_region`, the fourth schema_with property, has no tuple of its own -- schema_with
# derives it as `REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA[version] > 0`, so mirroring that map
# below mirrors the property. Listing a nonexistent TRANSITION_REGION_* tuple here would abort the
# run on the guard immediately after.
_MIRRORED = (
    "TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS",
    "GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS",
    "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS",
    "V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS",
    "V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS",
)
_missing = [n for n in _MIRRORED if not re.search(rf'^{n}\s*=', s, re.M)]
if _missing:
    raise SystemExit(
        "drill: property tuple(s) not found in observation.py: " + ", ".join(_missing) + ". "
        "The mirror cannot copy a membership it cannot read, and an unmirrored property routes "
        "the synthetic schema down a branch its reference does not take."
    )
s += f'''

# --- appended by scripts/schema_rotation_drill.sh; not part of the repo ---
_DRILL_OUTGOING = {_out}
for _dn in {_MIRRORED!r}:
    _dt = globals()[_dn]
    if _DRILL_OUTGOING in _dt:
        globals()[_dn] = _dt + (OBSERVATION_SCHEMA_VERSION_V5_DRILL,)
REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA = dict(REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA) | {{
    OBSERVATION_SCHEMA_VERSION_V5_DRILL: REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA[_DRILL_OUTGOING],
}}
'''
open(p, "w").write(s)
# Persisted so the preconditions and the scorer assert the premise against the schema actually
# replaced, rather than re-deriving it and risking a different answer.
open(f"{wt}/DRILL_OUTGOING.txt", "w").write(_out + "\n")

q = f"{wt}/src/pokezero/showdown.py"
t = open(q).read()
m = re.search(r'^REPLAY_OBSERVATION_SPECS_BY_SCHEMA[^=]*=\s*\{', t, re.M)
if m is None:
    raise SystemExit(
        "drill: could not locate the REPLAY_OBSERVATION_SPECS_BY_SCHEMA literal in showdown.py. "
        "An unregistered synthetic schema makes every spec lookup for it raise, and the whole "
        "breakage set would be drill artifact rather than evidence."
    )
# Stamp the synthetic spec with its OWN version. Mapping v5-drill to V4_REPLAY_OBSERVATION_SPEC
# left the table incoherent -- a spec stamped v4 reachable under the v5-drill key -- and
# `test_spec_for_schema_is_loud_on_unknown_versions` caught it, correctly, as
# "'pokezero.observation.v4' != 'pokezero.observation.v5-drill'". That was the DRILL's defect
# masquerading as a surviving instance of the class it is meant to detect: an instrument that
# manufactures the failure it reports.
import os as _os
_shape = _os.environ.get("DRILL_SHAPE", "identical")
# Clone the OUTGOING DEFAULT's spec, not v4's. The naming convention is exact and checked below:
# OBSERVATION_SCHEMA_VERSION_<S>  <->  <S>_REPLAY_OBSERVATION_SPEC, for every supported <S>.
_out_spec = f"{_suffix}_REPLAY_OBSERVATION_SPEC"
if not re.search(rf'^{_out_spec}\s*=', t, re.M):
    raise SystemExit(
        f"drill: the outgoing default is {_out}, but showdown.py has no {_out_spec} to clone. "
        "The synthetic schema must have the shape of the schema it replaces, or a breakage cannot "
        "be attributed to NAMING rather than to layout."
    )
if _shape == "differ":
    # One fewer numeric column than the OUTGOING DEFAULT. Small enough that nothing structural
    # changes, large enough that any consumer carrying the old width mismatches loudly.
    _spec_expr = ("_dc_replace(\n"
                  f"        {_out_spec},\n"
                  "        schema_version=OBSERVATION_SCHEMA_VERSION_V5_DRILL,\n"
                  f"        numeric_feature_count={_out_spec}.numeric_feature_count - 1,\n"
                  "    )")
else:
    _spec_expr = ("_dc_replace(\n"
                  f"        {_out_spec},\n"
                  "        schema_version=OBSERVATION_SCHEMA_VERSION_V5_DRILL,\n"
                  "    )")
t = t[:m.end()] + f"\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL: {_spec_expr}," + t[m.end():]
t = t.replace("REPLAY_OBSERVATION_SPECS_BY_SCHEMA", "REPLAY_OBSERVATION_SPECS_BY_SCHEMA", 1)
if "_dc_replace" not in t.split("REPLAY_OBSERVATION_SPECS_BY_SCHEMA")[0]:
    t = re.sub(r'^(import .*)$', r'from dataclasses import replace as _dc_replace\n\1', t, count=1, flags=re.M)
# Register the synthetic schema in EVERY schema-keyed table, not just the spec map. Two
# earlier drill runs charged `KeyError: 'pokezero.observation.v5-drill'` to the codebase when
# it was the drill failing to register its own schema in the census maps -- an instrument
# manufacturing the failure it reports, for the third time. Any table keyed by schema is part
# of "registering a schema"; missing one makes every consumer of it look like a defect.
# Appended AFTER each table is built, not inside the literal: referencing a dict from within
# its own construction is a NameError, which the first cut of this hit immediately.
_tables = ("_MINIMUM_CATEGORICAL_CENSUS_BY_SCHEMA", "_MINIMUM_NUMERIC_CENSUS_BY_SCHEMA")

for table in _tables:
    if not re.search(rf'^{table}\b', t, re.M):
        raise SystemExit(f"drill: schema-keyed table {table} not found -- registration incomplete")
# In `differ` mode the numeric census FLOOR must narrow with the width. Copying v4's floor
# verbatim gave the synthetic schema 131 numeric columns against a floor of 132, so EVERY
# default-spec encode raised at the validator ("requires exactly 132 numeric columns, got 131")
# and the entire breakage set was drill artifact -- the instrument manufacturing its own failure
# for the fourth time. This is the defect the header listed as unknown.
#
# Keyed off the OUTGOING DEFAULT, not v4, for the same reason as the spec and the property tuples:
# copying v4's floor onto a synthetic clone of v2.2 gives the schema a floor from a different
# layout, and every default-spec encode fails the validator on a mismatch the drill itself created.
_floor_delta = {"_MINIMUM_NUMERIC_CENSUS_BY_SCHEMA": (" - 1" if _shape == "differ" else "")}
_reg = "\n\n" + "\n".join(
    f"{tb} = dict({tb}) | {{OBSERVATION_SCHEMA_VERSION_V5_DRILL: "
    f"{tb}[{_out}]{_floor_delta.get(tb, '')}}}"
    for tb in _tables
) + "\n"
# Anchor on the DEFINITION and brace-match it. A first cut used the last textual reference to
# the table name, which lives inside a function 3000 lines below, and spliced the registration
# into the middle of an f-string -- SyntaxError. "Last mention" is not "end of definition".
_ends = []
for tb in _tables:
    md = re.search(rf'^{tb}[^=]*=\s*\{{', t, re.M)
    i = md.end(); depth = 1
    while depth:
        if t[i] == "{": depth += 1
        elif t[i] == "}": depth -= 1
        i += 1
    _ends.append(i)
_eol = t.index("\n", max(_ends)) + 1
t = t[:_eol] + _reg + t[_eol:]
if "OBSERVATION_SCHEMA_VERSION_V5_DRILL" not in t.split("REPLAY_OBSERVATION_SPECS_BY_SCHEMA")[0]:
    # The OUTGOING DEFAULT's constant must come across too: the census registration above is keyed
    # by it, and showdown.py only imports the versions it happens to mention. Injecting v4 alone
    # left a NameError waiting for any tree whose default is not v4.
    t = re.sub(r'^(from \.observation import \()',
               "\\1\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL,\n    OBSERVATION_SCHEMA_VERSION_V4,\n    "
               + _out + ",", t, count=1, flags=re.M)
open(q, "w").write(t)

# `_EXPORTABLE_TABLE_SCHEMAS` in engine_env.py, keyed by SHORT names ("v2.2", "v3", "v4") rather
# than full version strings -- which is exactly why every name-based search for schema-keyed
# structures missed it. Unregistered, `_schema_for_encoder_tables` raises ValueError and the whole
# of EngineEnvTest fails: 10 of the 19 "unexpected breakages" in the first scored run were this one
# table. Fifth instance of the instrument manufacturing the failure it reports, and the second to
# reach a verdict.
# Register the synthetic schema in EVERY literal container of schema names classified REGISTER,
# driven by tests/data/schema_drill_schema_containers.txt rather than by a hardcoded list. The
# hardcoded version registered `_EXPORTABLE_TABLE_SCHEMAS` and missed the exporter CLI's argparse
# `choices` eleven lines away in another file -- so `engine_env` shelled out to that script, argparse
# refused 'v5-drill', and ten EngineEnvTest failures were reported as surviving conflations. Sixth
# instance of that class; driving it from the census is what stops there being a seventh.
#
# PARTIAL containers are deliberately left alone: `vocab_shift_probe` compares one named PAIR of
# schemas, and a third member changes what it does rather than widening it.
_census = f"{wt}/tests/data/schema_drill_schema_containers.txt"
_reg_targets = []
for _raw in open(_census):
    _l = _raw.strip()
    if not _l or _l.startswith("#"):
        continue
    # 3 OR 4 fields: the classification file gained a variable-name column when the key was made
    # finer (F3). Unpacking exactly three aborted the injection with "too many values to unpack" --
    # loudly, which is why it was caught in seconds rather than scored.
    _parts = _l.split()
    _kind, _path, _members = _parts[0], _parts[1], _parts[2]
    if _kind == "REGISTER":
        _reg_targets.append((_path, sorted(_members.split(","))))
# DEDUPED on (file, members). The substitution below is FILE-WIDE -- it rewrites every occurrence of
# the anchor literal in one pass -- so running it once per ROW double-inserted when two containers in
# one file share a member set. Live, not hypothetical: neural_cli.py has two argparse tuples with the
# same choices (train at :467, iterate at :1951) and each ended up with `"v5-drill", "v5-drill"`.
# Caught by review; my own check missed it because `grep -c` counts LINES, and both insertions were on
# the same line.
_seen_reg = set()
_reg_targets = [
    t for t in _reg_targets
    if not (( t[0], tuple(t[1]) ) in _seen_reg or _seen_reg.add(( t[0], tuple(t[1]) )))
]
if not _reg_targets:
    raise SystemExit(
        "drill: the container census lists no REGISTER rows. Either the file is empty or its format "
        "changed; registering nothing would leave every schema-keyed container blind to the "
        "synthetic schema."
    )
# The CLI-choice DICT needs a key AND a value; the generic member-append would write a bare string
# into a dict literal, which is a SyntaxError. Handled first and explicitly, because a silent syntax
# error in showdown.py would make every import fail and the entire breakage set drill artifact.
_cli = f"{wt}/src/pokezero/showdown.py"
_ct = open(_cli).read()
_m_cli = re.search(r'^OBSERVATION_SCHEMA_CLI_CHOICES[^=]*=\s*\{(.*?)\n\}', _ct, re.S | re.M)
if _m_cli is None:
    raise SystemExit(
        "drill: could not locate the OBSERVATION_SCHEMA_CLI_CHOICES dict in showdown.py. "
        "`observation_schema_version_from_choice` raises on an unknown key, so leaving the "
        "synthetic schema out of it makes every consumer of the encoder-table exporter fail on the "
        "drill's own omission -- which is how ten EngineEnvTest failures were reported as surviving "
        "conflations."
    )
_ct = (_ct[:_m_cli.end(1)]
       + '\n    "v5-drill": OBSERVATION_SCHEMA_VERSION_V5_DRILL,'
       + _ct[_m_cli.end(1):])
open(_cli, "w").write(_ct)
print('  registered "v5-drill" in OBSERVATION_SCHEMA_CLI_CHOICES (dict: key AND value)')

# CONSTANT-valued containers: `{OBSERVATION_SCHEMA_VERSION_V2_2, ..._V3, ..._V4}` in the exporter's
# `main()`, and `{..._V4: "v4", ...}` in the lattice. Both are ad-hoc, inline in a function body, and
# NOT module-level property tuples -- the census docstring used to claim containers of constants were
# covered by the property mirroring, which was false for exactly this shape. Ninth and eighth
# instances of the class. Each also needs the V5_DRILL constant IMPORTED, or the file NameErrors at
# call time and every consumer fails on the drill's omission.
# Anchored on the CONTAINER, matched as a whole multi-member literal -- NOT on a bare constant name.
# The first cut anchored on "OBSERVATION_SCHEMA_VERSION_V4," and used replace(..., 1), which matched
# the IMPORT LIST 329 lines above the set. The print said "registered", the set was untouched, and
# EngineEnvTest failed again on the same `parser.error`. A print is not a verification, so each
# registration is now ASSERTED below by re-parsing the file and checking the synthetic constant is
# inside the intended container.
for _cpath, _pat, _sub in (
    # `if schema_version not in { V2_2, V3, V4 }:` in main()
    ("scripts/export_encoder_tables.py",
     re.compile(r'(not in \{\s*\n(?:\s*OBSERVATION_SCHEMA_VERSION_V[0-9_]+,\s*\n)+)(\s*\}\s*:)'),
     r'\1        OBSERVATION_SCHEMA_VERSION_V5_DRILL,\n\2'),
    # `_EXPORTER_SCHEMA_CHOICES = { V4: "v4", ... }`
    ("src/pokezero/mcts_eval/lattice.py",
     re.compile(r'(_EXPORTER_SCHEMA_CHOICES\s*=\s*\{\s*\n(?:\s*OBSERVATION_SCHEMA_VERSION_V[0-9_]+:\s*"[^"]*",\s*\n)+)(\s*\})'),
     r'\1        OBSERVATION_SCHEMA_VERSION_V5_DRILL: "v5-drill",\n\2'),
):
    _cf = f"{wt}/{_cpath}"
    _cx = open(_cf).read()
    _cx2, _nsub = _pat.subn(_sub, _cx, count=1)
    if _nsub != 1:
        raise SystemExit(
            f"drill: could not locate the schema container in {_cpath} to register the synthetic "
            "schema in. It gates a per-schema path and raises on an unknown schema, so leaving it "
            "unregistered makes its consumers fail on the drill's own omission -- which is how ten "
            "EngineEnvTest failures were reported as surviving conflations, twice."
        )
    _cx = _cx2
    # The constant must be IMPORTED there, or the module NameErrors at call time -- worse than
    # leaving it unregistered, because the failure then looks like a codebase defect. It did: the
    # first cut tested `^\s*OBSERVATION_SCHEMA_VERSION_V5_DRILL,\s*$`, which the set insertion two
    # lines above had ALREADY satisfied, so the import was skipped and the exporter died on
    # "NameError: name 'OBSERVATION_SCHEMA_VERSION_V5_DRILL' is not defined". Asked of the AST now,
    # which distinguishes an imported name from a name that merely appears.
    def _imports_drill(src: str) -> bool:
        for _n in ast.walk(ast.parse(src)):
            if isinstance(_n, ast.ImportFrom) and any(
                a.name == "OBSERVATION_SCHEMA_VERSION_V5_DRILL" for a in _n.names
            ):
                return True
        return False

    if not _imports_drill(_cx):
        # Anchor on the V4 constant INSIDE an ImportFrom's name list, located by AST position rather
        # than by pattern, so it cannot match the container edit made above.
        _ins = None
        for _n in ast.walk(ast.parse(_cx)):
            if isinstance(_n, ast.ImportFrom):
                for _a in _n.names:
                    if _a.name.startswith("OBSERVATION_SCHEMA_VERSION_V"):
                        _ins = _a
        if _ins is None:
            raise SystemExit(
                f"drill: {_cpath} imports no per-version schema constant, so there is no import list "
                "to add the synthetic one to. Registering the container without the import would "
                "NameError at call time."
            )
        _lines = _cx.split("\n")
        _li = _ins.lineno - 1
        _indent = _lines[_li][: len(_lines[_li]) - len(_lines[_li].lstrip())]
        _lines.insert(_li + 1, f"{_indent}OBSERVATION_SCHEMA_VERSION_V5_DRILL,")
        _cx = "\n".join(_lines)
        if not _imports_drill(_cx):
            raise SystemExit(
                f"drill: tried to add the synthetic constant to {_cpath}'s imports and the AST still "
                "does not see it imported. Refusing to proceed on an announcement."
            )
    open(_cf, "w").write(_cx)
    # ASSERT the registration landed where it was meant to, by re-parsing.
    _tree = ast.parse(_cx)
    _ok = any(
        isinstance(n, (ast.Set, ast.Dict))
        and any(
            isinstance(e, ast.Name) and e.id == "OBSERVATION_SCHEMA_VERSION_V5_DRILL"
            for e in (n.elts if isinstance(n, ast.Set) else [k for k in n.keys if k])
        )
        for n in ast.walk(_tree)
    )
    if not _ok:
        raise SystemExit(
            f"drill: wrote the synthetic constant into {_cpath} but it is NOT a member of any set or "
            "dict there -- the edit landed somewhere else (the first attempt landed in the import "
            "list). Registration must be verified, not announced."
        )
    print(f"  registered the synthetic schema in {_cpath} (constant-valued container, re-parsed)")

_registered = 0
for _path, _members in _reg_targets:
    if _path == "src/pokezero/showdown.py" and _members[0].startswith("v"):
        continue  # the CLI-choices dict, handled above
    if _members[0].startswith("OBSERVATION_SCHEMA_VERSION_"):
        continue  # constant-valued, handled above
    _f = f"{wt}/{_path}"
    _txt = open(_f).read()
    _short = not _members[0].startswith("pokezero.observation.")
    _new = "v5-drill" if _short else "pokezero.observation.v5-drill"
    # Anchor on the LAST member of the container, matched with its quotes, and insert after it. The
    # members are literals, so this cannot splice into an expression.
    _last = _members[-1]
    _pat = re.compile(r'(["\']){}\1'.format(re.escape(_last)))
    _n_before = len(_pat.findall(_txt))
    if _n_before == 0:
        raise SystemExit(
            f"drill: could not find the literal {_last!r} in {_path} to anchor the synthetic "
            "schema after. The census says this container must be registered; if its spelling "
            "changed, re-run scripts/schema_container_census.py --check and reclassify."
        )
    _txt = _pat.sub(lambda m: f'{m.group(0)}, "{_new}"', _txt)
    open(_f, "w").write(_txt)
    _registered += _n_before
print(f'  registered the synthetic schema in {len(_reg_targets)} classified container(s), '
      f'{_registered} literal site(s)')
print("  synthetic v5-drill schema injected; "
      + ("default LEFT in place (control arm)" if _control else "default rotated to it"))
PY
# `cat` to an empty file then `python3 emptyfile` exits 0, so an unchecked write turns disk-full
# into a silent "injection succeeded, nothing injected" -- the same trap the snapshot copy guards.
[ -s "$INJ" ] || { echo "ABORT: the injection script snapshot is empty"; exit 10; }
python3 "$INJ" "$WT" || { echo "FAILED to inject the drill schema"; exit 3; }

# PRECONDITION -1: every literal container of schema NAMES is classified. Run against the PRISTINE
# repo, not the injected worktree, so the census sees the code as written rather than as mutated.
#
# Six times a schema-keyed structure went unregistered and the drill charged the resulting failures
# to the codebase -- SUPPORTED, GROUPED_LAYOUT, FEATURE_PACK, V2_1_LINEAGE, the two census maps,
# `_EXPORTABLE_TABLE_SCHEMAS`, and the exporter CLI's argparse `choices`. Each was found by running
# the full drill, reading a stack trace, and adding the one table it named. That loop does not
# converge: after each fix the instrument still had no way to say what it had missed, so a clean run
# only proved that the tables I had thought of were registered. This enumerates them instead, and an
# unclassified container aborts BEFORE anything is scored.
"$VENV" "$REPO/scripts/schema_container_census.py" --check || {
  echo "ABORT: a schema-name container is unclassified, so the drill cannot know whether to"
  echo "       register its synthetic schema in it. An unregistered container makes its consumers"
  echo "       fail on the drill's own omission, which reads as a surviving defect."
  exit 15
}

# PRECONDITION 0: no unlisted schema identity gate exists. A line-oriented grep used to do this
# and caught 2 of 7 evasive forms -- it missed `!=`, `in (literal tuple)`, string literals, dict
# keys, `is`, and match/case, and TWO live gates evaded it while it claimed none existed. The
# scanner is AST-based, so the syntactic form is irrelevant.
"$VENV" "$WT/scripts/schema_identity_gate_scan.py" || {
  echo "ABORT: an unlisted schema identity gate exists, so a breakage cannot be attributed to"
  echo "       NAMING -- the synthetic schema would route down a wrong branch."
  exit 13
}

# PRECONDITION 1: the rotation actually happened. The injection edits are regex-based and their
# results were never checked, so a reformatted source line would silently no-op and the drill
# would score an UNROTATED tree as clean.
"$VENV" - <<'PRE' || exit 8
import sys
sys.path.insert(0, __import__("os").environ["DRILL_WT"] + "/src")
from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION as v,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS as SUP,
    schema_with,
)
assert v.endswith("v5-drill"), f"default is {v!r}, not the synthetic schema -- injection no-oped"
assert SUP[-1] == v, f"synthetic schema is not LAST in SUPPORTED ({SUP[-1]!r}); schema_with() " \
                     "iterates reversed() and would never return it, leaving every " \
                     "property-selector site inert under the drill"
# The migration's own vocabulary must resolve TO the synthetic schema, or every site the migration
# moved onto the property selector is inert under the drill. Queried by the OUTGOING DEFAULT's
# property profile, not by `feature_pack=True`: that hardcoded v4's distinguishing property, so on
# a v2.2 tree it asserted that a query v2.2 does not even satisfy returns the v2.2 clone -- which
# fails for the right reason by luck, or passes while testing nothing, depending on the tree.
import os as _os
from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION_V5_DRILL,
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS as FP,
    GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS as GL,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS as TM,
    REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA as TTC,
)
_outgoing = open(_os.environ["DRILL_WT"] + "/DRILL_OUTGOING.txt").read().strip()
import pokezero.observation as _obs
_out_v = getattr(_obs, _outgoing)
_profile = {
    "transition_region": TTC[_out_v] > 0,
    "turn_merged": _out_v in TM,
    "grouped_layout": _out_v in GL,
    "feature_pack": _out_v in FP,
}
_got = schema_with(**_profile)
assert _got == v, (
    f"schema_with(**{_profile}) -> {_got!r}, not the synthetic schema. That profile is the "
    f"OUTGOING default's ({_outgoing}), which the synthetic schema mirrors, so it must resolve to "
    "the synthetic schema now that it is newest. Otherwise the migration's own vocabulary is "
    "untested by this run."
)
print(f"  default is now: {v}")
print(f"  outgoing default: {_outgoing} -> profile {_profile}")
print(f"  schema_with(**outgoing profile) -> {_got}")
PRE

# PRECONDITION 2: the synthetic schema is spec- and routing-equivalent to THE SCHEMA BEING ROTATED
# AWAY FROM. This is the drill's central premise -- "shape-identical, so every breakage is a naming
# failure" -- and it has now been false twice.
#
# First: V2_1_LINEAGE went unregistered and two numeric cells differed.
#
# Second, and far worse, this precondition itself compared against v4 while the injection also
# cloned v4. On the pre-rotation tree the outgoing default is v2.2, so it verified "the v4 clone
# equals v4" -- true by construction, and silent about the fact that the rotation moved the
# default's shape from 155/51/151 to 132/41/23. The premise was false and its own guard passed. A
# guard whose reference is the thing it is comparing cannot fail.
#
# The reference is now the outgoing default, read from the file the injection wrote rather than
# re-derived, so the two halves cannot disagree about which schema was replaced. In `differ` mode
# the numeric width is EXPECTED to be one narrower -- that is the arm's whole purpose -- so that
# one field is compared against the intended delta instead of for equality.
"$VENV" - <<'PRE' || exit 9
import os, sys
sys.path.insert(0, os.environ["DRILL_WT"] + "/src")
import pokezero.observation as _obs
from pokezero.observation import OBSERVATION_SCHEMA_VERSION as drill
from pokezero.showdown import observation_spec_for_schema
_outgoing = open(os.environ["DRILL_WT"] + "/DRILL_OUTGOING.txt").read().strip()
ref = getattr(_obs, _outgoing)
_shape = os.environ.get("DRILL_SHAPE", "identical")
a, b = observation_spec_for_schema(ref), observation_spec_for_schema(drill)
fields = ("token_count", "categorical_feature_count", "numeric_feature_count",
          "transition_token_count", "opponent_tendency_stats_token_count")
_expected_delta = {"numeric_feature_count": -1} if _shape == "differ" else {}
bad = [
    (f, getattr(a, f), getattr(b, f))
    for f in fields
    if getattr(b, f) != getattr(a, f) + _expected_delta.get(f, 0)
]
if bad:
    print(f"  SPEC MISMATCH {_outgoing} vs synthetic (shape={_shape}):", bad)
    print("  Expected the synthetic schema to have the OUTGOING default's shape"
          + (" with numeric_feature_count one narrower." if _shape == "differ" else "."))
    sys.exit(9)
from pokezero.observation import (
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS, GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS, V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS,
    V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS, REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA,
)
tuples = {
    "FEATURE_PACK": FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    "GROUPED_LAYOUT": GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
    "TURN_MERGED": TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
    "V2_1_LINEAGE": V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS,
    "V3_PROJECTION": V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS,
}
skew = [n for n, t in tuples.items() if (ref in t) != (drill in t)]
if REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA.get(ref) != REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA.get(drill):
    skew.append("REPLAY_TRANSITION_TOKEN_COUNTS")
if skew:
    print(f"  MEMBERSHIP SKEW {_outgoing} vs synthetic in:", skew)
    print(f"  The synthetic schema is not equivalent to {_outgoing}, so a breakage cannot be attributed")
    print("  to NAMING. Register it in these structures before trusting any verdict.")
    sys.exit(9)
print(f"  premise verified: synthetic schema is spec- and membership-equivalent to {_outgoing}"
      + (" (numeric width one narrower, as `differ` intends)" if _shape == "differ" else ""))
PRE

# PRECONDITION 3: the arm actually DISCRIMINATES, observed through pytest rather than inferred
# from the data model. PRECONDITION 2 proves the specs differ as intended; it does not prove the
# test suite can SEE that difference. This runs a throwaway canary that hardcodes the outgoing
# default's numeric width -- a deliberate shape conflation, the exact defect the differ arm exists
# to detect -- through the real collection/import/assert path:
#
#     identical  -> canary MUST PASS  (shape unchanged, so a width pin is untouched)
#     differ     -> canary MUST FAIL  (one column narrower, so a width pin breaks)
#
# This is the acceptance test the differ arm was missing. It was previously unconstructible: while
# the injection cloned v4, the arm-distinguishing width was v4's 132 against a v2.2 baseline of
# 155, so a 132 canary failed at baseline, landed in the subtracted baseline set, and vanished from
# BOTH arms -- which is why differ could be run and still prove nothing. Cloning the outgoing
# default makes the discriminating width 155, which passes at baseline and under identical and
# fails only under differ.
#
# The expected width is read from the UNMUTATED $REPO, not from the mutated worktree, so the canary
# cannot be satisfied by the same edit it is meant to detect. The file is deleted immediately after,
# before anything is scored, so it can never appear as a breakage.
_canary="$WT/tests/test__drill_shape_canary.py"
"$VENV" - "$REPO" "$WT" <<'PRE' || exit 11
import os, re, subprocess, sys
repo, wt = sys.argv[1], sys.argv[2]
outgoing = open(wt + "/DRILL_OUTGOING.txt").read().strip()
# Width of the outgoing default, read from the pristine checkout.
code = (
    "import sys; sys.path.insert(0, %r + '/src');"
    "import pokezero.observation as o;"
    "from pokezero.showdown import observation_spec_for_schema as s;"
    "print(s(getattr(o, %r)).numeric_feature_count)" % (repo, outgoing)
)
w = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
if w.returncode != 0:
    print("  CANARY SETUP FAILED reading the pristine width:", w.stderr.strip()[-400:]); sys.exit(11)
width = int(w.stdout.strip())
shape = os.environ.get("DRILL_SHAPE", "identical")
path = wt + "/tests/test__drill_shape_canary.py"
open(path, "w").write(
    "import unittest\n"
    "from pokezero.showdown import DEFAULT_REPLAY_OBSERVATION_SPEC as D\n"
    "class DrillShapeCanary(unittest.TestCase):\n"
    "    def test_the_default_numeric_width_is_unchanged(self):\n"
    f"        self.assertEqual(D.numeric_feature_count, {width})\n"
)
env = dict(os.environ, PYTHONPATH=wt + "/src")
r = subprocess.run([sys.executable, "-m", "pytest", path, "-q", "-p", "no:randomly"],
                   capture_output=True, text=True, env=env)
os.remove(path)  # before ANYTHING is scored
passed = r.returncode == 0
want_pass = shape != "differ"
print(f"  shape canary: pinned width {width} (from the pristine tree), "
      f"shape={shape}, canary {'passed' if passed else 'failed'}")
if passed != want_pass:
    print(f"  CANARY DID NOT DISCRIMINATE: under shape={shape} the canary was expected to "
          f"{'PASS' if want_pass else 'FAIL'} and it did not.")
    print("  " + ("The identical arm changed the default's shape, so a breakage in it cannot be"
                  " attributed to naming." if want_pass else
                  "The differ arm did NOT change the default's observable shape, so it covers"
                  " nothing that the identical arm does not -- the shape half is UNCOVERED."))
    print(r.stdout[-1200:])
    sys.exit(11)
print(f"  arm discriminates: a width pin {'survives' if want_pass else 'breaks'} under shape={shape}")
PRE
rm -f "$_canary"

# Validate the INSTRUMENT without paying for a verdict. Everything above is the injection and its
# preconditions -- the part that decides whether a breakage can be attributed to naming at all --
# and it runs in seconds, while the two suites below take ~22 minutes. Every instrument defect this
# script has recorded was in the part above. Exits 0 having produced NO verdict, and says so, so a
# preconditions-only run can never be quoted as a PASS.
if [ "${DRILL_STOP_AFTER_PRECONDITIONS:-0}" = "1" ]; then
  echo "== preconditions only (DRILL_STOP_AFTER_PRECONDITIONS=1) =="
  echo "NO VERDICT: the injection and its preconditions passed; no test was run and nothing was"
  echo "            scored. This is an instrument check, not a result. Re-run without the flag."
  echo "Worktree kept for inspection: $WT"
  exit 0
fi

find "$WT/tests" -name __pycache__ -exec rm -rf {} + 2>/dev/null
PYTHONPATH="$WT/src" "$VENV" -m pytest $(drill_targets "$WT") -q -p no:randomly > "$WT/DRILL.txt" 2>&1

echo "== result =="
tail -1 "$WT/DRILL.txt"
# The HUMAN-readable summary must use the same normaliser as the scorer. This line kept both
# defects the scorer had already had fixed: it grepped ^FAILED only (blind to SUBFAILED) and
# assumed absolute paths -- so it printed 14 directly beneath a summary line saying 18, with every
# entry retaining an unstripped "FAILED tests/" prefix. A reader trusts the readable half.
echo "-- raw failures by file (pre-subtraction; see the scored verdict below) --"
grep -E '^(FAILED|SUBFAILED)' "$WT/DRILL.txt" | _norm_id | sed 's|::.*||' | sort | uniq -c | sort -rn
_raw=$(grep -cE '^(FAILED|SUBFAILED)' "$WT/DRILL.txt" || true)
echo "   raw total: $_raw  (must equal the summary's failure count; cross-checked below)"
# BASELINE. Without it the drill counts pre-existing failures as breakages: the first scored
# run charged `test_roll_enumeration_scope` to the rotation when it actually fails on the 3.11
# f-string defect in c153, and would have kept charging it forever. A breakage is a test that
# passes UNROTATED and fails ROTATED -- anything else is noise being attributed to this class.
# BASELINE COMMIT: HEAD, and that is CORRECT for this drill -- an earlier warning here claimed
# otherwise and was wrong.
#
# This drill answers one question: "if the default rotates to a schema NOTHING names, does
# anything except the class-(iii) readers notice?" That question is asked FROM the current state,
# so the baseline must be the current state. A pre-rotation baseline cannot answer it, and it
# cannot even be constructed coherently: the class-(iii) tests assert which schema currently
# holds the slot, so on any tree whose default differs they are already red, get subtracted, and
# report as "pins no longer pinning" -- the drill's own contradiction signal, fired by the
# measurement setup rather than by a defect.
#
# The DIFFERENT question -- "did rotating v2.2 -> v4 break anything?" -- is not this drill's job
# and must not be inferred from it. It is answered by running the full suite on the rotated tree
# and on the pre-rotation commit and diffing the failure SETS. That was done: byte-identical to
# main's 8 pre-existing failures. Keep the two separate; conflating them is how the earlier
# warning came to demand an impossible baseline.
BASE_REF="${DRILL_BASELINE_REF:-HEAD}"
echo "== baseline: $BASE_REF, same interpreter, default NOT rotated =="
# Derived FROM $WT, never a sibling guess. The old default was "$WT/../schema-drill-baseline",
# which force-removed an unrelated real directory when the drill was invoked as documented.
BASE="${DRILL_BASELINE:-${WT%/}-baseline}"
# Reusable: two full suites in one job exceeds most runners' patience, and the baseline only
# changes when HEAD does. Set DRILL_BASELINE_REUSE=1 with a baseline already computed at this
# same commit AND the same scope. Both are recorded: a baseline from a different commit, or
# from `fast` reused under `full`, would silently subtract the wrong set -- and a wrong
# subtraction is invisible, it just makes the residue look smaller than it is.
if [ "${DRILL_BASELINE_REUSE:-0}" = "1" ] && [ -f "$BASE/BASE.sha" ] \
   && [ "$(cat "$BASE/BASE.sha")" = "$(git -C "$REPO" rev-parse "$BASE_REF") ${DRILL_SCOPE:-full} ${DRILL_SHAPE:-identical} $("$VENV" -c 'import sys;print(sys.version.split()[0])')" ]; then
  echo "  reusing baseline at $(cat "$BASE/BASE.sha" | cut -c1-8)"
  # The STABLE intersection, persisted by the fresh path -- NOT run 2 alone. `BASE.txt` is a copy of
  # BASE.2.txt, so subtracting it here reinstated the single-run baseline and with it the defect this
  # drill closed: a flake in run 2 earns a permanent excuse for every reused run afterwards. And the
  # reuse path printed no "UNSTABLE ... NOT subtracted" warning, so the excuse was silent. Worse, the
  # header calls reuse the NORMAL path ("two full suites in one job exceeds most runners' patience"),
  # so the closed defect lived on the path most runs take.
  if [ ! -s "$BASE/baseline.stable.txt" ]; then
    echo "ABORT: the cached baseline predates the stable-intersection format. Re-run without"
    echo "       DRILL_BASELINE_REUSE=1 to regenerate it; a single-run baseline lets a flake"
    echo "       earn a permanent excuse."
    exit 11
  fi
  cp "$BASE/baseline.stable.txt" "$WT/baseline.txt"
  if [ -s "$BASE/baseline.unstable.txt" ]; then
    echo "  UNSTABLE across the two cached baseline runs, NOT subtracted:"
    sed 's/^/    /' "$BASE/baseline.unstable.txt"
  fi
  echo "  baseline failures (NOT attributable to the rotation): $(wc -l < "$WT/baseline.txt" | tr -d ' ')"
  SKIP_BASELINE=1
fi
if [ "${SKIP_BASELINE:-0}" != "1" ]; then
git -C "$REPO" worktree remove --force "$BASE" 2>/dev/null
git -C "$REPO" worktree add -q --detach "$BASE" "$BASE_REF" || exit 3
find "$BASE/tests" -name __pycache__ -exec rm -rf {} + 2>/dev/null

# TWO baseline runs, and only the INTERSECTION is subtracted. A single run let a flaky test earn a
# permanent excuse: `test_bench_apply_reverse_returns_positive_rate` failed in one baseline, passed
# 5/5 in isolation and passed in the rotated arm -- so it silently entered the blind set, where any
# genuine breakage of that test would have been excused with no signal. A test that fails in only
# one of two identical runs is unstable, not pre-existing, and is DROPPED from the excuse list so a
# real breakage still surfaces.
for _run in 1 2; do
  find "$BASE/tests" -name __pycache__ -exec rm -rf {} + 2>/dev/null
  PYTHONPATH="$BASE/src" "$VENV" -m pytest $(drill_targets "$BASE") -q -p no:randomly \
    > "$BASE/BASE.$_run.txt" 2>&1
  grep -E '^(FAILED|SUBFAILED)' "$BASE/BASE.$_run.txt" | _norm_id | sort -u > "$BASE/b$_run.txt"
done
cp "$BASE/BASE.2.txt" "$BASE/BASE.txt"          # the summary guards read the second run
comm -12 "$BASE/b1.txt" "$BASE/b2.txt" > "$WT/baseline.txt"     # stable failures only
# PERSISTED, so a reused run subtracts the same stable set rather than run 2 alone.
cp "$WT/baseline.txt" "$BASE/baseline.stable.txt"
comm -3 "$BASE/b1.txt" "$BASE/b2.txt" > "$BASE/baseline.unstable.txt"
_unstable=$(comm -3 "$BASE/b1.txt" "$BASE/b2.txt" | grep -c . || true)
echo "  baseline failures, STABLE across two runs (subtracted): $(wc -l < "$WT/baseline.txt" | tr -d ' ')"
if [ "${_unstable:-0}" -gt 0 ]; then
  echo "  UNSTABLE across the two baseline runs, NOT subtracted ($_unstable) -- a flake must not"
  echo "  become a permanent excuse; a real breakage in these would otherwise be invisible:"
  comm -3 "$BASE/b1.txt" "$BASE/b2.txt" | sed 's/^/    /' | head
fi
# Stamp written AFTER both runs complete, and covering everything that changes the baseline. The
# first version wrote it BEFORE pytest, so an interrupted baseline was reusable and
# indistinguishable from a finished one; and it omitted DRILL_SHAPE and the interpreter.
echo "$(git -C "$REPO" rev-parse "$BASE_REF") ${DRILL_SCOPE:-full} ${DRILL_SHAPE:-identical} $("$VENV" -c 'import sys;print(sys.version.split()[0])')" > "$BASE/BASE.sha"
fi

# The summary's failure count MUST equal the number of ids we scored. They disagreed by 4 once
# (18 vs 14, pytest-subtests) and the shortfall was silent. This makes any future missed bucket
# -- a new pytest reporter prefix, a plugin -- a hard failure instead of a quiet undercount.
# DENOMINATOR. Without this, a suite killed partway prints a summary line, passes the
# "has a summary" guard, and scores PASS over a fraction of the tests -- verified: a synthetic
# "18 failed, 20 passed" summary produced "PASS: the class is dead" and exit 0. All seven pins
# sort at or before test_turn_merged_encode.py, so any truncation past that point looks clean.
_tot() { # passed+failed+skipped from a pytest summary line
  local f="$1" p x s2
  p=$(grep -oE '[0-9]+ passed' "$f" | tail -1 | grep -oE '[0-9]+' || echo 0)
  x=$(grep -oE '[0-9]+ failed' "$f" | tail -1 | grep -oE '[0-9]+' || echo 0)
  s2=$(grep -oE '[0-9]+ skipped' "$f" | tail -1 | grep -oE '[0-9]+' || echo 0)
  echo $(( ${p:-0} + ${x:-0} + ${s2:-0} ))
}
_rot_tot=$(_tot "$WT/DRILL.txt"); _base_tot=$(_tot "$BASE/BASE.txt")
echo "  denominator: rotated=$_rot_tot baseline=$_base_tot"
if [ "$_rot_tot" -lt 100 ] || [ "$_base_tot" -lt 100 ]; then
  echo "ABORT: a run collected only $_rot_tot / $_base_tot tests -- that is not this suite."
  exit 11
fi
_skew=$(( _rot_tot > _base_tot ? _rot_tot - _base_tot : _base_tot - _rot_tot ))
if [ "$_skew" -gt 5 ]; then
  echo "ABORT: rotated and baseline denominators differ by $_skew ($_rot_tot vs $_base_tot)."
  echo "       One run measured a different suite; the subtraction is meaningless."
  exit 11
fi

_summary_failed=$(grep -oE '^[0-9]+ failed' "$WT/DRILL.txt" | head -1 | grep -oE '[0-9]+' || echo 0)
_scored=$(grep -cE '^(FAILED|SUBFAILED)' "$WT/DRILL.txt" || true)
# XPASS: an `unittest.expectedFailure` that unexpectedly PASSES is not in "N failed" and emits no
# scored line -- a genuinely invisible bucket (n=1 today, test_interaction_registry). Under a
# rotation an xfail flipping to xpass means the thing it documented as broken now works, which is
# a behaviour change the drill must not swallow.
# Counted as a QUANTITY, not as a line count. `grep -c` counts matching LINES, so a summary line
# reading "3 xpassed" scored 1 and this note would have printed "1 xpass marker(s)" directly above
# evidence of three -- the same readable-summary-contradicts-the-number defect as D15, reintroduced
# in the fix for a different one. Takes the max of the summary quantity and the per-line marker
# count because which of the two a run emits depends on the reporter's verbosity.
_xpass_sum=$(grep -oE '[0-9]+ xpassed' "$WT/DRILL.txt" | head -1 | grep -oE '[0-9]+' || echo 0)
_xpass_lines=$(grep -cE '^XPASS' "$WT/DRILL.txt" || true)
_xpass=$(( ${_xpass_sum:-0} > ${_xpass_lines:-0} ? ${_xpass_sum:-0} : ${_xpass_lines:-0} ))
if [ "${_xpass:-0}" -gt 0 ]; then
  echo "  NOTE: $_xpass xpass(es) in the rotated run -- an expectedFailure now passes."
  grep -E '^XPASS|[0-9]+ xpassed' "$WT/DRILL.txt" | sed 's/^/    /' | head -5
fi
if [ "${_summary_failed:-0}" -ne "${_scored:-0}" ]; then
  echo "ABORT: pytest reported ${_summary_failed} failures but only ${_scored} were scored."
  echo "       A reporter bucket is unaccounted for; the score would be an undercount."
  exit 6
fi

# Rubric comes from the WORKTREE (committed HEAD), not the live tree. Reading it from $REPO let
# the pass condition and the code under test come from different revisions: checking out an
# unrelated branch mid-run made the file vanish, `expected` became 0, and all 8 legitimate
# class-(iii) breakages reported as UNEXPECTED. Predicted by review before it happened.
# `differ` gets its OWN pass condition. The class-(iii) set means "asserts which schema is
# default"; under a WIDTH change every site that legitimately NAMES a width is also a legitimate
# break, which is a different population entirely. Sharing one rubric would either fail the run on
# correct behaviour or -- worse -- let a real shape conflation hide behind an entry admitted for
# the naming run.
if [ "${DRILL_SHAPE:-identical}" = "differ" ]; then
  EXPECTED="$WT/tests/data/schema_drill_expected_breakages_shape.txt"
else
  EXPECTED="$WT/tests/data/schema_drill_expected_breakages.txt"
fi
if [ ! -f "$EXPECTED" ]; then
  echo "ABORT: no expected-breakages file at $EXPECTED -- there is no pass condition to score"
  echo "       against, and scoring without one reports every legitimate breakage as unexpected."
  exit 7
fi
# Collection ERRORs and a non-zero-but-no-FAILED run are both "the suite did not measure what
# it claims". Scoring only ^FAILED made a module-level import failure read as "6 pins no longer
# pinning" rather than "nothing ran" -- the exact symptom this drill hit twice.
# The CONTROL logs are in this list too. They were not, so a control run that died before pytest
# printed a summary yielded an EMPTY failure set -- which subtracts nothing and silently turns
# off the arm whose whole job is to prevent over-attribution.
for f in "$WT/DRILL.txt" "$BASE/BASE.txt"; do
  [ -f "$f" ] || continue
  if ! grep -qE '^[0-9]+ (passed|failed)' "$f"; then
    echo "ABORT: $f has no pytest summary line -- the run did not complete."; exit 4
  fi
done
# ERRORs are subtracted like FAILEDs, not treated as an automatic abort. The first cut aborted
# on ANY error and immediately fired on the pre-existing `fallback_replay` errors, which surface
# as ERROR rather than FAILED under the fast scope -- a guard against "the run measured nothing"
# that instead blocked every run. A NEW error (one absent from the baseline) is still fatal:
# that is the module-level-import case, where the suite silently stops and the score would read
# as "pins no longer pinning".
# Normalise to the bare test id and compare against BOTH baseline errors and baseline failures.
# A test already broken in the baseline may change failure MODE under the rotation -- the seven
# pre-existing `fallback_replay` tests FAIL unrotated and ERROR under a shape-differing schema,
# because the breakage moves into setUpClass. Same broken tests, different pytest bucket; not
# something the rotation caused. Two bugs were here at once: the sed also assumed a leading
# `/tests/` and produced "ERROR ERROR ..." for relative paths.
_norm() { sed -E 's#^(ERROR|FAILED)[[:space:]]+##; s#^.*/tests/##; s#^tests/##' "$1" | sort -u; }
grep -E '^(ERROR|FAILED|SUBFAILED)' "$BASE/BASE.txt" > "$WT/base_broken.raw" || true
grep '^ERROR ' "$WT/DRILL.txt" > "$WT/rot_err.raw" || true
_norm "$WT/base_broken.raw" > "$WT/base_broken.txt"
_norm "$WT/rot_err.raw" > "$WT/rot_err.txt"
NEW_ERR=$(comm -13 "$WT/base_broken.txt" "$WT/rot_err.txt")
if [ -n "$NEW_ERR" ]; then
  echo "ABORT: the rotated run has ERRORs the baseline does not -- the suite did not measure:"
  printf '%s\n' "$NEW_ERR" | sed 's/^/  /' | head
  exit 4
fi
# SUBFAILED too. pytest-subtests reports a failing subtest as `SUBFAILED(param) <id>` and counts
# it in the summary's "N failed", but it does NOT emit a `FAILED` line. Grepping ^FAILED alone hid
# 4 real failures behind an 18-vs-14 discrepancy between the summary and the scored set -- the
# eighth instrument defect here, and another that hides failures rather than inventing them.
grep -E '^(FAILED|SUBFAILED)' "$WT/DRILL.txt" | _norm_id | sort -u > "$WT/rotated.txt"
# Empty normalised output means the normaliser broke, not that the tree is clean -- the sed
# delimiter bug produced exactly that and it scored as a PASS. Placed AFTER rotated.txt is
# written: the first cut of this guard sat six lines too early and tested a file that did not
# exist yet, so it aborted every run. A guard in the wrong place is still a bug, just a noisy
# one instead of a silent one.
if grep -q '^FAILED' "$WT/DRILL.txt" && [ ! -s "$WT/rotated.txt" ]; then
  echo "ABORT: the run has FAILED lines but normalisation produced nothing -- scorer is broken."
  exit 5
fi
# Subtract the baseline, THEN the source-mutation artifacts. Both are "not attributable to the
# default moving", but for different reasons, and they are kept in different files so a real
# conflation cannot hide behind the word "expected".
ARTIFACTS="$WT/tests/data/schema_drill_source_mutation_artifacts.txt"
grep -vE '^\s*(#|$)' "$ARTIFACTS" 2>/dev/null | sort -u > "$WT/artifacts.txt" || : > "$WT/artifacts.txt"

# ============================ CONTROL ARM: register, do NOT rotate ============================
# The drill's question is "what breaks because the DEFAULT MOVED". Its injection does two things at
# once: it ADDS a schema and it MOVES the default. Subtracting only the baseline attributes both to
# the rotation, and the first scored run showed exactly what that costs -- among 19 "surviving
# instances of the class" were tests whose subject is the schema INVENTORY and which break, wholly
# correctly, on any new schema:
#
#   test_observation.py::...::test_the_supported_window_and_the_legacy_refusal   asserts the
#       SUPPORTED tuple equals a specific 5-tuple; the drill appends a 6th.
#   test_observation_spec_v4.py::...::test_v4_is_supported_...                   asserts
#       SUPPORTED[-1] is v4; the drill appends after it.
#   test_schema_with_selector.py::...::test_newest_first_is_deliberate_and_pinned  asserts
#       schema_with() returns the newest; the synthetic schema IS the newest.
#
# Those are not conflations. A test that pins the supported inventory SHOULD break when a schema is
# added -- that is a human acknowledging a new schema, and it is the opposite of a defect. Charging
# them to the rotation inflates the verdict and buries whatever real conflations sit beside them.
#
# So: a third arm that runs the identical injection with the rotation SKIPPED. Anything red there is
# caused by ADDING a schema. attributable = rotated - baseline - control.
CTRL="${WT%/}-control"
if [ "${DRILL_SKIP_CONTROL:-0}" = "1" ]; then
  : > "$WT/control.txt"
  echo "  CONTROL ARM SKIPPED (DRILL_SKIP_CONTROL=1) -- inventory pins will be charged to the"
  echo "  rotation and the unexpected count is an OVER-estimate. Not a quotable verdict."
else
  echo "== control: schema registered, default NOT rotated =="
  git -C "$REPO" worktree remove --force "$CTRL" 2>/dev/null || true
  git -C "$REPO" worktree add -q --detach "$CTRL" HEAD || exit 3
  DRILL_WT="$CTRL" DRILL_NO_ROTATE=1 python3 "$INJ" "$CTRL" || exit 3
  find "$CTRL/tests" -name __pycache__ -exec rm -rf {} + 2>/dev/null
  # TWICE, INTERSECTED -- exactly as the baseline does, and for the identical reason. This arm
  # subtracts from the scored set, so a test that fails ONCE here earns a permanent excuse and a
  # genuine naming breakage disappears with no trace. The baseline was fixed for that (D6/D12: "a
  # single run let a flaky test earn a permanent excuse"); this arm was added later and shipped with
  # the same defect, demonstrated by review with a probe that failed in the rotated arm, passed both
  # baselines, and failed once in control -- it vanished from the verdict.
  for _crun in 1 2; do
    # PER RUN, as the baseline does. Clearing once before both let the second reuse the first's
    # bytecode, CORRELATING the two runs -- and their intersection is SUBTRACTED, so correlated runs
    # can excuse a real rotation breakage. That is the direction that matters.
    find "$CTRL/tests" -name __pycache__ -exec rm -rf {} + 2>/dev/null
    PYTHONPATH="$CTRL/src" "$VENV" -m pytest $(drill_targets "$CTRL") -q -p no:randomly \
      > "$CTRL/CONTROL.$_crun.txt" 2>&1 || true
    # CHECKED HERE, where the log exists and $CTRL is assigned. A control run that died before pytest
    # printed a summary yields an EMPTY failure set, which subtracts nothing and silently turns off the
    # arm whose whole job is to prevent over-attribution -- and with both runs dead, `_cunstable` is 0
    # too, so the run reports "control failures beyond baseline: 0" with no warning at all.
    if ! grep -qE '^[0-9]+ (passed|failed)' "$CTRL/CONTROL.$_crun.txt"; then
      echo "ABORT: $CTRL/CONTROL.$_crun.txt has no pytest summary line -- control run $_crun did not"
      echo "       complete, and an empty control set silently disables the arm."
      tail -5 "$CTRL/CONTROL.$_crun.txt" | sed 's/^/  /'
      exit 4
    fi
    grep -E '^(FAILED|SUBFAILED)' "$CTRL/CONTROL.$_crun.txt" | _norm_id | sort -u > "$CTRL/c$_crun.txt"
  done
  cp "$CTRL/CONTROL.1.txt" "$CTRL/CONTROL.txt"   # the log the reader is pointed at
  comm -12 "$CTRL/c1.txt" "$CTRL/c2.txt" > "$WT/control_raw.txt"
  comm -3 "$CTRL/c1.txt" "$CTRL/c2.txt" > "$CTRL/control.unstable.txt"
  _cunstable=$(grep -c . "$CTRL/control.unstable.txt" || true)
  comm -23 "$WT/control_raw.txt" "$WT/baseline.txt" > "$WT/control.txt"
  echo "  control failures beyond baseline (caused by ADDING a schema, not by rotating): $(wc -l < "$WT/control.txt" | tr -d ' ')"
  sed 's/^/    /' "$WT/control.txt" | head -12
  [ "$(wc -l < "$WT/control.txt")" -gt 12 ] && echo "    ... and $(( $(wc -l < "$WT/control.txt") - 12 )) more"
  if [ "${_cunstable:-0}" -gt 0 ]; then
    echo "  UNSTABLE across the two control runs, NOT subtracted ($_cunstable) -- a flake must not"
    echo "  excuse a rotation breakage:"
    sed 's/^/    /' "$CTRL/control.unstable.txt"
  fi
fi

# ===================== NATIVE-SCHEMA EXCLUSION, derived by CAUSE not by name =====================
# The Rust leaf encoder dispatches on schema version in COMPILED code
# (rust/pokezero-search/src/encoder.rs: "unsupported observation layout schema"). This drill edits
# Python source in a git worktree and never rebuilds the crate, so a synthetic schema can NEVER
# satisfy that match -- no amount of registration reaches it.
#
# That is a hard scope limit of the instrument, and it cost four rounds of chasing to establish:
# registering `_EXPORTABLE_TABLE_SCHEMAS`, then the exporter's argparse `choices`, then
# `OBSERVATION_SCHEMA_CLI_CHOICES`, then the exporter's inline validation set and the lattice's
# `_EXPORTER_SCHEMA_CHOICES` -- each fix revealing the next gate down the same path, until the last
# one turned out to be compiled.
#
# Excluded by CAUSE: a test is excluded only if its own failure output carries that Rust message.
# A name list would go stale silently and would let an unrelated failure in the same test hide behind
# the excuse; matching the message means the exclusion is justified per-test, per-run, by evidence in
# the log. Printed in full below, never silently dropped.
"$VENV" - "$WT" <<'NATIVE' > "$WT/native_schema.txt"
import re, sys
wt = sys.argv[1]
text = open(f"{wt}/DRILL.txt", errors="replace").read()
MSG = "unsupported observation layout schema"
# The message must appear in a RAISED-EXCEPTION line, not anywhere in the section body. A plain
# `MSG in body` substring test was defeated by review with a two-line probe: a test failing for a
# pure NAMING reason whose assertion message merely CONTAINED the string was excluded and vanished
# from the verdict. Any test whose captured log, docstring, or assertion text mentions the error --
# including a test ABOUT the error -- would have been silently subtracted.
#
# pytest prefixes traceback and exception lines in a FAILURES section with "E ". Requiring the match
# there means the exclusion is justified by the exception the test actually raised.
# ANCHORED IMMEDIATELY AFTER `ValueError:`, which is the only thing the Rust bridge raises
# (rust/pokezero-search/src/encoder.rs raises PyValueError, surfacing as ValueError). Two weaker rules
# were tried and both were defeated by review:
#
#   `MSG in body`                     any mention anywhere -- assertion message, captured log, a test
#                                     ABOUT the error.
#   `^E\s+<Type>(Error|Exception):`   `AssertionError` satisfies the type token, and REAL pytest
#                                     renders a custom assertion message on ONE line as
#                                     `E   AssertionError: 1 != 2 : <msg>`. So assertion messages were
#                                     still excluded, and the outcome depended on pytest's diff
#                                     line-wrapping rather than on cause.
#
# My kill-confirm for the second rule reported "assertion-message probe -> kept" and was WRONG,
# because I tested it against a two-line render I invented instead of pytest's actual one-line output.
# The probe data was the defect, not the rule under test.
#
# Anchored means `ValueError: <MSG>` with nothing between -- an assertion message reaches MSG only
# after `AssertionError: ... : `, so it cannot match.
E_LINE = re.compile(r'^E\s+ValueError:\s*' + re.escape(MSG), re.M)
# The splitter accepts headers containing SPACES. pytest-subtests emits
# `___ Cls.test (i=1) ___`, which `(\S+)` cannot match, so those bodies MERGED INTO THE PRECEDING
# SECTION -- and an innocent naming breakage was excluded because the NEXT section raised the Rust
# error. 77 files under tests/ use subTest, including test_engine_env.py, which is this exclusion's
# own target population. The captured name is normalised: the ` (i=1)` subtest tail and any `[param]`
# tail are stripped so it matches the `Class.test` form the filter compares against.
parts = re.split(r'\n_+ (.+?) _+\n', text)
ids = []
for i in range(1, len(parts), 2):
    # The tail is CONVERTED, never stripped. pytest's header is `Cls.test (i=1)`; `_norm_id`
    # produces `file.py::Cls::test[i=1]`. Stripping it made the exclusion TEST-granular while the
    # scored set is SUBTEST-granular, so a Rust ValueError in ONE subtest excused an innocent naming
    # breakage in a DIFFERENT subtest of the same test -- and that contradicts `_norm_id`'s own
    # stated invariant, which keeps the param precisely so one failing subtest cannot excuse its
    # siblings. Demonstrated by review against real pytest output; latent in v8 only because no
    # exclusion there had a sibling.
    name = re.sub(r'\s*\((.*)\)$', r'[\1]', parts[i].strip())
    body = parts[i + 1] if i + 1 < len(parts) else ""
    if E_LINE.search(body):
        ids.append(name)
print("\n".join(sorted(set(ids))))
NATIVE
_native_n=$(grep -c . "$WT/native_schema.txt" || true)
if [ "${_native_n:-0}" -gt 0 ]; then
  echo "  NATIVE-SCHEMA scope limit: $_native_n test(s) fail inside the COMPILED Rust encoder's"
  echo "  schema dispatch, which this drill cannot register into. Excluded by cause (their own"
  echo "  failure output carries the message), not by name:"
  sed 's/^/    /' "$WT/native_schema.txt" | head -12
  [ "$_native_n" -gt 12 ] && echo "    ... and $(( _native_n - 12 )) more"
fi

comm -23 "$WT/rotated.txt" "$WT/baseline.txt" > "$WT/attributable_pre_control.txt"
comm -23 "$WT/attributable_pre_control.txt" "$WT/control.txt" > "$WT/attributable_pre_native.txt"
# Section headers carry only `Class.test`, so the exclusion is resolved to FULL ids before use, and
# an AMBIGUOUS tail is a hard abort rather than a guess. The first cut compared on `Class.test` and
# discarded the file, so a same-named test in an UNRELATED file was excluded too -- demonstrated by
# review: a probe at `test_zz_unrelated_file.py::EngineEnvTest::test_observation_validates_...` was
# dropped by an exclusion earned in `test_engine_env.py`.
"$VENV" - "$WT" <<'FILTER' > "$WT/attributable.txt"
import sys
wt = sys.argv[1]
native = {l.strip() for l in open(f"{wt}/native_schema.txt") if l.strip()}


def tail(i):
    # Keeps the [param] tail, for the same reason native.py now keeps it: dropping it collapses every
    # subtest of a test onto one key, so excluding one excluded all of them.
    core, _, param = i.partition("[")
    bits = core.split("::")
    base = ".".join(bits[-2:]) if len(bits) >= 2 else core
    return f"{base}[{param}" if param else base


# Ambiguity is checked against the WHOLE rotated set, because a same-named test in another file is a
# collision wherever it sits. The FILTER, though, must be applied to `attributable_pre_native.txt` --
# the set already reduced by the baseline and control subtractions. An earlier version of this block
# read `rotated.txt` for both, which would have bypassed those two subtractions entirely and inflated
# the verdict. Two different inputs, deliberately.
rotated = [l.rstrip("\n") for l in open(f"{wt}/rotated.txt") if l.strip()]
candidates = [l.rstrip("\n") for l in open(f"{wt}/attributable_pre_native.txt") if l.strip()]
by_tail = {}
for i in rotated:
    by_tail.setdefault(tail(i), set()).add(i)
ambiguous = {t: sorted(v) for t, v in by_tail.items() if t in native and len(v) > 1}
if ambiguous:
    sys.stderr.write(
        "drill: a native-schema exclusion is AMBIGUOUS -- the same Class.test exists in more than "
        "one file, so excluding by tail would drop a test that never raised the Rust error:\n"
    )
    for t, v in sorted(ambiguous.items()):
        sys.stderr.write(f"  {t} -> {v}\n")
    sys.exit(16)
excluded_full = {f for t, v in by_tail.items() if t in native for f in v}
for i in candidates:
    if i not in excluded_full:
        print(i)
FILTER
# Artifact matching is TEST-granular on purpose, while ids are SUBTEST-granular. A source-mutation
# artifact (a citation-pinning test broken by the drill editing source) breaks in every subtest, so
# listing 24 rows would be noise -- but the excusal has to be deliberate, not a side effect of
# collapsing ids. Strip the `[param]` suffix only for this comparison; the baseline subtraction
# above stays subtest-granular so a flake in one subtest cannot excuse its siblings.
sed -E 's#\[[^]]*\]$##' "$WT/attributable.txt" | paste -d'\t' - "$WT/attributable.txt" \
  | sort > "$WT/attr_pairs.txt"
join -t$'\t' -v1 -1 1 -2 1 "$WT/attr_pairs.txt" <(sort "$WT/artifacts.txt") \
  | cut -f2 | sort -u > "$WT/actual.txt"
_art=$(comm -12 <(sed -E 's#\[[^]]*\]$##' "$WT/attributable.txt" | sort -u) <(sort "$WT/artifacts.txt") | grep -c . || true)
[ "${_art:-0}" -gt 0 ] && echo "  source-mutation artifact tests subtracted: $_art (all their subtests)"
grep -vE '^\s*(#|$)' "$EXPECTED" | sort -u > "$WT/expected.txt"
# Present-but-all-comments is the same as absent, and combined with a no-op rotation it scored a
# PASS. `-f` was not enough.
[ -s "$WT/expected.txt" ] || { echo "ABORT: the expected-breakages file has no entries."; exit 7; }
# Every rubric row must be a BARE id -- no spaces. A row carrying a reason suffix, or a stray
# comment tail, can never match a normalised breakage, and the failure presents as "expected but did
# not break", i.e. as a dead pin rather than as a malformed rubric.
if grep -qE '[[:space:]]' "$WT/expected.txt"; then
  echo "ABORT: rubric rows must be bare test ids with no whitespace. Offending rows:"
  grep -nE '[[:space:]]' "$WT/expected.txt" | sed 's/^/  /'
  exit 7
fi

UNEXPECTED=$(comm -23 "$WT/actual.txt" "$WT/expected.txt")
MISSING=$(comm -13 "$WT/actual.txt" "$WT/expected.txt")
NU=$(printf '%s' "$UNEXPECTED" | grep -c . || true)
NM=$(printf '%s' "$MISSING" | grep -c . || true)

echo
echo "expected (class-iii, must break): $(wc -l < "$WT/expected.txt" | tr -d ' ')"
echo "actual breakages:                 $(wc -l < "$WT/actual.txt" | tr -d ' ')"
echo
if [ "$NU" -gt 0 ]; then
  echo "UNEXPECTED BREAKAGES ($NU) -- surviving instances of the class, each earns a ledger row:"
  printf '%s\n' "$UNEXPECTED" | sed 's/^/  /'
fi
if [ "$NM" -gt 0 ]; then
  # A class-(iii) test that did NOT break is worse than an unexpected one: it means the pin on
  # the default's identity has stopped pinning, so the next rotation goes unnoticed.
  echo "EXPECTED-BUT-DID-NOT-BREAK ($NM) -- these pins are no longer pinning:"
  printf '%s\n' "$MISSING" | sed 's/^/  /'
fi
echo
echo "-- BLIND SET: $(wc -l < "$WT/baseline.txt" | tr -d ' ') stable baseline failures + $(wc -l < "$WT/artifacts.txt" | tr -d ' ') source-mutation artifacts --"
echo "   A genuine breakage of any of these would be subtracted and invisible. Printed rather"
echo "   than implied, because an unexamined excuse list is where a real defect hides."
sed 's/^/     /' "$WT/baseline.txt" | head -12
[ "$(wc -l < "$WT/baseline.txt")" -gt 12 ] && echo "     ... and $(( $(wc -l < "$WT/baseline.txt") - 12 )) more"
# The ARTIFACTS half, printed too. It was counted in the header above and then never listed -- and
# it is the half that excuses BY NAME at test granularity across every subtest, so it is the one
# where an unexamined excuse would hide a real defect. The stated rationale applied to the other
# list only.
echo "   source-mutation artifacts (excused by name):"
sed 's/^/     /' "$WT/artifacts.txt"
echo
_mode="scope=${DRILL_SCOPE:-full} shape=${DRILL_SHAPE:-identical} rotated=$_rot_tot baseline=$_base_tot"
if [ "$NU" -eq 0 ] && [ "$NM" -eq 0 ]; then
  if [ "${DRILL_SHAPE:-identical}" != "identical" ] || [ "${DRILL_SCOPE:-full}" != "full" ]; then
    # A `fast` or `differ` transcript used to be byte-indistinguishable from a full identical
    # run, including the words "The class is dead". Labels that live only in source comments are
    # not labels.
    echo "INCONCLUSIVE ($_mode): breakage set matches, but this configuration is NOT the stop"
    echo "  condition. fast cannot see a new breakage in a file that has never broken; differ is"
    echo "  a SCOPED arm. Re-run with no DRILL_SCOPE before quoting a result; DRILL_SHAPE=differ is"
    echo "  verified and has its own rubric, but 'fast' cannot see a new breakage in an unlisted file."
    exit 12
  fi
  echo "PASS ($_mode): the breakage set is EXACTLY the class-(iii) readers."
  echo "  SCOPE OF THIS CLAIM: no test reads the default's VERSION where it means a specific one."
  echo "  It does NOT cover shape-dependent conflations. The synthetic schema is shape-identical to"
  echo "  the OUTGOING DEFAULT by design, so a site hardcoding that schema's width sees no change"
  echo "  in this arm and its conflation goes undetected here. DRILL_SHAPE=differ is the arm for"
  echo "  that half; until it has been run and its result stated, the shape half is UNCOVERED."
fi
echo "Full log: $WT/DRILL.txt"
exit $(( NU + NM == 0 ? 0 : 1 ))
