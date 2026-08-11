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
# Exit 0 = the class is dead. Nonzero = breakages beyond the legitimate readers; see the diff.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT="${1:-/tmp/schema-v5-drill}"
VENV="$REPO/.venv/bin/python"

echo "== drill: synthetic v5 rotation =="
git -C "$REPO" worktree remove --force "$WT" 2>/dev/null
git -C "$REPO" worktree add -q --detach "$WT" HEAD || exit 3
cd "$WT" || exit 3

python3 - "$WT" <<'PY'
import re, sys
wt = sys.argv[1]
p = f"{wt}/src/pokezero/observation.py"
s = open(p).read()

# A v5 that is byte-identical to v4 in shape. The drill is about NAMING, not about layout: any
# shape difference would make breakages ambiguous between "reached the default" and "assumed a
# layout". Identical shape means every breakage is unambiguously a naming failure.
anchor = re.search(r'^OBSERVATION_SCHEMA_VERSION_V4\s*=.*$', s, re.M)
s = s[:anchor.end()] + '\nOBSERVATION_SCHEMA_VERSION_V5_DRILL = "pokezero.observation.v5-drill"' + s[anchor.end():]
s = re.sub(r'^(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS\s*=\s*\()', r'\1\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL,', s, count=1, flags=re.M)
s = re.sub(r'^(GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS\s*=\s*\()', r'\1\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL,', s, count=1, flags=re.M)
s = s.replace("FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS = (OBSERVATION_SCHEMA_VERSION_V4,)",
              "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS = (OBSERVATION_SCHEMA_VERSION_V4, OBSERVATION_SCHEMA_VERSION_V5_DRILL)")
s = re.sub(r'^(\s*)OBSERVATION_SCHEMA_VERSION_V4: (V4_TRANSITION_TOKEN_COUNT,)$',
           r'\1OBSERVATION_SCHEMA_VERSION_V4: \2\n\1OBSERVATION_SCHEMA_VERSION_V5_DRILL: \2', s, count=1, flags=re.M)
s = s.replace("OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V4",
              "OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSION_V5_DRILL")
open(p, "w").write(s)

q = f"{wt}/src/pokezero/showdown.py"
t = open(q).read()
m = re.search(r'^REPLAY_OBSERVATION_SPECS_BY_SCHEMA[^=]*=\s*\{', t, re.M)
t = t[:m.end()] + "\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL: V4_REPLAY_OBSERVATION_SPEC," + t[m.end():]
if "OBSERVATION_SCHEMA_VERSION_V5_DRILL" not in t.split("REPLAY_OBSERVATION_SPECS_BY_SCHEMA")[0]:
    t = re.sub(r'^(from \.observation import \()', r'\1\n    OBSERVATION_SCHEMA_VERSION_V5_DRILL,', t, count=1, flags=re.M)
open(q, "w").write(t)
print("  synthetic v5-drill schema injected; default rotated to it")
PY
[ $? -eq 0 ] || { echo "FAILED to inject the drill schema"; exit 3; }

"$VENV" -c "import sys; sys.path.insert(0,'$WT/src'); from pokezero.observation import OBSERVATION_SCHEMA_VERSION as v; print('  default is now:', v)" || exit 3

find "$WT/tests" -name __pycache__ -exec rm -rf {} + 2>/dev/null
PYTHONPATH="$WT/src" "$VENV" -m pytest "$WT/tests" -q -p no:randomly \
  --ignore="$WT/tests/test_terminal_disposition_register.py" \
  --ignore="$WT/tests/test_unreachable_readjudication.py" \
  --ignore="$WT/tests/test_wide_seed_negative_census.py" > "$WT/DRILL.txt" 2>&1

echo "== result =="
tail -1 "$WT/DRILL.txt"
echo "-- breakages by file --"
grep '^FAILED' "$WT/DRILL.txt" | sed 's|.*/tests/||; s|::.*||' | sort | uniq -c | sort -rn
EXPECTED="$REPO/tests/data/schema_drill_expected_breakages.txt"
grep '^FAILED' "$WT/DRILL.txt" | sed 's|.*/tests/||' | sort -u > "$WT/actual.txt"
grep -vE '^\s*(#|$)' "$EXPECTED" | sort -u > "$WT/expected.txt"

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
[ "$NU" -eq 0 ] && [ "$NM" -eq 0 ] && echo "PASS: the breakage set is EXACTLY the class-(iii) readers. The class is dead."
echo "Full log: $WT/DRILL.txt"
exit $(( NU + NM == 0 ? 0 : 1 ))
