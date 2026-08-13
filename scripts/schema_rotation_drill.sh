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
#         DRILL_SHAPE=differ ...  EXPERIMENTAL, NOT SOUND YET. See the warning below.
#
# Why a shape-differing variant exists. The default synthetic schema is shape-IDENTICAL to v4 so
# that every breakage is unambiguously a NAMING failure. The cost of that choice is that no
# shape-dependent conflation can break under it -- and both real bugs this effort found
# (#1227 token_count, #1228 the feature widths) were shape bugs. A drill that structurally
# cannot reproduce its own motivating defects is not sufficient evidence on its own.
#
# STATUS: `differ` is CONSTRUCTED but UNVERIFIED. Its output must not be quoted as evidence until
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
# change. Its rubric is committed EMPTY for that reason -- populating it from an unverified run
# would launder a guess into a pin.
#
# So what remains is verification, not construction, and it needs a tree where the drill runs.
# The acceptance test is specific: a canary asserting `numeric_feature_count == 132` must break
# under the differ arm (131 columns there) and must not break under the identical arm. Until that
# is observed, the shape half of the class is UNCOVERED and the honest verdict says so.
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
  sed -E 's#[[:space:]]+-[[:space:]].*\$##; s#^(FAILED|ERROR)[[:space:]]+##; s#^SUBFAILED\(([^)]*)\)[[:space:]]+(.*)\$#\2[\1]#; s#^SUBFAILED[[:space:]]+##; s#(^|::)[^ ]*/tests/#\1#; s#^tests/##'
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
git -C "$REPO" worktree remove --force "$WT" 2>/dev/null
git -C "$REPO" worktree add -q --detach "$WT" HEAD || exit 3
cd "$WT" || exit 3
export DRILL_WT="$WT"

python3 - "$WT" <<'PY'
import pathlib, re, sys
wt = sys.argv[1]
p = f"{wt}/src/pokezero/observation.py"
s = open(p).read()

# A v5 that is byte-identical to v4 in shape. The drill is about NAMING, not about layout: any
# shape difference would make breakages ambiguous between "reached the default" and "assumed a
# layout". Identical shape means every breakage is unambiguously a naming failure.
anchor = re.search(r'^OBSERVATION_SCHEMA_VERSION_V4\s*=.*$', s, re.M)
s = s[:anchor.end()] + '\nOBSERVATION_SCHEMA_VERSION_V5_DRILL = "pokezero.observation.v5-drill"' + s[anchor.end():]
# Appended LAST, not first. `schema_with()` iterates `reversed(SUPPORTED)` to prefer the NEWEST
# match, so inserting the synthetic schema at the front made it the oldest and `schema_with()`
# never returned it -- every site the migration moved to the property selector was INERT under
# the drill, which is the migration's own vocabulary going untested by its acceptance check.
_m = re.search(r'^SUPPORTED_OBSERVATION_SCHEMA_VERSIONS\s*=\s*\((.*?)\)', s, re.S | re.M)
if _m is None:
    raise SystemExit("drill: could not locate SUPPORTED_OBSERVATION_SCHEMA_VERSIONS")
s = s[:_m.end(1)] + "\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL," + s[_m.end(1):]
s = re.sub(r'^(GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS\s*=\s*\()', r'\1\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL,', s, count=1, flags=re.M)
s = s.replace("FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS = (OBSERVATION_SCHEMA_VERSION_V4,)",
              "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS = (OBSERVATION_SCHEMA_VERSION_V4, OBSERVATION_SCHEMA_VERSION_V5_DRILL)")
# V2_1_LINEAGE was MISSED, and v4 is a member. It drives `schema_v2_1` in showdown.py (sub-HP
# fraction, PP-validity bits, pinned Tier-2 investment, defender identity), so omitting it made
# the synthetic schema encode DIFFERENTLY from v4 -- two numeric cells -- which falsifies the
# drill's central premise that only NAMING can break. Third defect from this same class; the
# encode-equivalence precondition below is what actually closes it.
_m21 = re.search(r'^V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS\s*=\s*\((.*?)\)', s, re.S | re.M)
if _m21 is None:
    raise SystemExit("drill: could not locate V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS")
s = s[:_m21.end(1)] + "\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL," + s[_m21.end(1):]
s = re.sub(r'^(\s*)OBSERVATION_SCHEMA_VERSION_V4: (V4_TRANSITION_TOKEN_COUNT,)$',
           r'\1OBSERVATION_SCHEMA_VERSION_V4: \2\n\1OBSERVATION_SCHEMA_VERSION_V5_DRILL: \2', s, count=1, flags=re.M)
s = s.replace("OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V4",
              "OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V5_DRILL")
open(p, "w").write(s)

q = f"{wt}/src/pokezero/showdown.py"
t = open(q).read()
m = re.search(r'^REPLAY_OBSERVATION_SPECS_BY_SCHEMA[^=]*=\s*\{', t, re.M)
# Stamp the synthetic spec with its OWN version. Mapping v5-drill to V4_REPLAY_OBSERVATION_SPEC
# left the table incoherent -- a spec stamped v4 reachable under the v5-drill key -- and
# `test_spec_for_schema_is_loud_on_unknown_versions` caught it, correctly, as
# "'pokezero.observation.v4' != 'pokezero.observation.v5-drill'". That was the DRILL's defect
# masquerading as a surviving instance of the class it is meant to detect: an instrument that
# manufactures the failure it reports.
import os as _os
_shape = _os.environ.get("DRILL_SHAPE", "identical")
if _shape == "differ":
    # One fewer numeric column than v4. Small enough that nothing structural changes, large
    # enough that any consumer carrying v4's width against this schema mismatches loudly.
    _spec_expr = ("_dc_replace(\n"
                  "        V4_REPLAY_OBSERVATION_SPEC,\n"
                  "        schema_version=OBSERVATION_SCHEMA_VERSION_V5_DRILL,\n"
                  "        numeric_feature_count=V4_REPLAY_OBSERVATION_SPEC.numeric_feature_count - 1,\n"
                  "    )")
else:
    _spec_expr = ("_dc_replace(\n"
                  "        V4_REPLAY_OBSERVATION_SPEC,\n"
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
_floor_delta = {"_MINIMUM_NUMERIC_CENSUS_BY_SCHEMA": (" - 1" if _shape == "differ" else "")}
_reg = "\n\n" + "\n".join(
    f"{tb} = dict({tb}) | {{OBSERVATION_SCHEMA_VERSION_V5_DRILL: "
    f"{tb}[OBSERVATION_SCHEMA_VERSION_V4]{_floor_delta.get(tb, '')}}}"
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
    t = re.sub(r'^(from \.observation import \()', r'\1\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL,\n    OBSERVATION_SCHEMA_VERSION_V4,', t, count=1, flags=re.M)
open(q, "w").write(t)
print("  synthetic v5-drill schema injected; default rotated to it")
PY
[ $? -eq 0 ] || { echo "FAILED to inject the drill schema"; exit 3; }

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
assert schema_with(feature_pack=True) == v, (
    f"schema_with(feature_pack=True) -> {schema_with(feature_pack=True)!r}, not the synthetic "
    "schema; the migration's own vocabulary is untested by this run"
)
print(f"  default is now: {v}")
print(f"  schema_with(feature_pack=True) -> {schema_with(feature_pack=True)}")
PRE

# PRECONDITION 2: the synthetic schema ENCODES IDENTICALLY to v4. This is the drill's central
# premise -- "shape-identical, so every breakage is a naming failure" -- and it was FALSE:
# V2_1_LINEAGE went unregistered and two numeric cells differed. Asserting the premise instead of
# documenting it closes the whole "unregistered schema-keyed table" class, which has produced
# THREE separate instrument defects. Any table the injection misses now shows up here as an
# encode difference, before a single test is scored.
"$VENV" - <<'PRE' || exit 9
import os, sys
sys.path.insert(0, os.environ["DRILL_WT"] + "/src")
from pokezero.observation import OBSERVATION_SCHEMA_VERSION as drill, OBSERVATION_SCHEMA_VERSION_V4 as v4
from pokezero.showdown import observation_spec_for_schema
a, b = observation_spec_for_schema(v4), observation_spec_for_schema(drill)
fields = ("token_count", "categorical_feature_count", "numeric_feature_count",
          "transition_token_count", "opponent_tendency_stats_token_count")
bad = [(f, getattr(a, f), getattr(b, f)) for f in fields if getattr(a, f) != getattr(b, f)]
if bad:
    print("  SPEC MISMATCH v4 vs synthetic:", bad); sys.exit(9)
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
skew = [n for n, t in tuples.items() if (v4 in t) != (drill in t)]
if REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA.get(v4) != REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA.get(drill):
    skew.append("REPLAY_TRANSITION_TOKEN_COUNTS")
if skew:
    print("  MEMBERSHIP SKEW v4 vs synthetic in:", skew)
    print("  The synthetic schema is not equivalent to v4, so a breakage cannot be attributed")
    print("  to NAMING. Register it in these structures before trusting any verdict.")
    sys.exit(9)
print("  premise verified: synthetic schema is spec- and membership-equivalent to v4")
PRE

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
_xpass=$(grep -coE '^XPASS|[0-9]+ xpassed' "$WT/DRILL.txt" || true)
if [ "${_xpass:-0}" -gt 0 ]; then
  echo "  NOTE: $_xpass xpass marker(s) in the rotated run -- an expectedFailure now passes."
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
for f in "$WT/DRILL.txt" "$BASE/BASE.txt"; do
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
comm -23 "$WT/rotated.txt" "$WT/baseline.txt" > "$WT/attributable.txt"
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
    echo "  known-unsound. Re-run with no DRILL_SCOPE/DRILL_SHAPE before quoting a result."
    exit 12
  fi
  echo "PASS ($_mode): the breakage set is EXACTLY the class-(iii) readers."
  echo "  SCOPE OF THIS CLAIM: no test reads the default's VERSION where it means a specific one."
  echo "  It does NOT cover shape-dependent conflations -- the synthetic schema is shape-identical"
  echo "  to v4 by design, so a site asserting a v4 width passes both arms. Pair with a shape probe."
fi
echo "Full log: $WT/DRILL.txt"
exit $(( NU + NM == 0 ? 0 : 1 ))
