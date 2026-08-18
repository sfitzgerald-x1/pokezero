#!/usr/bin/env python3
"""Mutation battery for the rollout-leaf witness, its guards and its shard schema.

WHY THIS IS A FILE AND NOT A PARAGRAPH. Round 2 flagged that the previous revision's
"19 mutants, 19 KILLED" was not reproducible: the count shipped without the list, so
it was unfalsifiable prose. `scripts/mutate_zero_heal_guard.py` is the precedent that
does it properly -- the mutant table lives in the repo, the run is recorded as a JSON
artifact, and a test holds the artifact to the harness that produced it. This is the
same shape applied to the surface #1271 adds.

THE CLASSIFIER IS VALIDATED BEFORE ITS COUNTS ARE TRUSTED, because the obvious
implementation is wrong in the flattering direction: `pytest` exits 1 on a COLLECTION
error, so "non-zero return code" classifies a syntax error as KILLED and a battery of
mutants that never ran reports as a clean sweep. Three controls therefore have to read
`DID NOT RUN` and are asserted to:

  * `_control_syntax_error`  -- an unparseable production module;
  * `_control_deleted_killer` -- the killer test module removed;
  * `_control_hang`          -- an unbounded sleep at import (subprocess timeout).

Two more prove the classifier can produce its other verdicts at all:

  * `_control_null`     -- no edit. Must read SURVIVED.
  * `_control_positive` -- a one-line break with a known killer. Must read KILLED.

KILLED requires, together: a non-zero return code, at least one reported test FAILURE,
`errors == 0`, and at least one `FAILED <nodeid>` line naming the test that died. A
run that exits non-zero with zero failures is a DID NOT RUN, not a kill.

MUTANTS ARE WRITTEN INTO THE TREE, not supplied by `PYTHONPATH`. `tests/conftest.py`
MOVES its own `src` to the front of `sys.path` (a move, not an insert-if-absent), so a
`PYTHONPATH`-supplied mutant is overridden and appears to survive without ever having
been loaded. Every run therefore prints the RESOLVED module path from inside the
subprocess and the harness refuses any run whose resolved path is not this tree's.

The tree is proven clean afterwards by re-reading and hashing every target file, not
by `git status`.

Usage:
    scripts/mutate_rollout_leaf_witness.py \
        --venv-python .venv/bin/python \
        --json reports/artifacts/rollout_leaf_witness_mutation_battery.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "pokezero" / "engine_search.py"
BRIDGE = ROOT / "src" / "pokezero" / "foulplay_bridge.py"

#: The modules whose tests are the KILLERS. Both, always: the witness lives in
#: `engine_search` and its last hop lives in `foulplay_bridge`, and the finding this
#: battery exists for is that the two were tested to different depths.
KILLERS = (
    "tests/test_rollout_model_priors.py",
    "tests/test_mcts_acceptance_report.py",
)

#: Printed from INSIDE the subprocess so the harness can prove which copy of the
#: module the killers actually imported.
RESOLVED_PROBE = (
    "import pokezero.engine_search as _es, pokezero.foulplay_bridge as _fb; "
    "print('RESOLVED_ENGINE=' + _es.__file__); "
    "print('RESOLVED_BRIDGE=' + _fb.__file__)"
)


class InstrumentFailure(RuntimeError):
    """Never a verdict. Always a non-zero exit for the whole battery.

    Borrowed verbatim from `scripts/c157_volatile_fail_conjunct_battery.py`: an
    instrument that cannot tell whether it ran must not emit a number.
    """


# --------------------------------------------------------------------------- #
# Anchors, hoisted so each is unique and the table below stays readable.
# --------------------------------------------------------------------------- #

WITNESS_CALL = (
    "        require_rollout_leaf_witness(metadata, rollout_leaf_eval=rollout_leaf_eval)\n"
)
BOUNDARY_CALL = "    require_banked_shard_witness(payload)\n"
POLICY_STATS_LINE = '                "policy_stats": self.policy_stats,\n'
ENGINE_STATS_RETURN = "    return policy.stats.to_dict()\n"
ENCODE_SKIP_REPORT_ENTRY = (
    '        "rollout_encode_skipped": "rollout_encode_skipped",\n'
)
# The bare field line appears in BOTH `ROLLOUT_LEAF_WITNESS_FIELDS` and
# `ROLLOUT_LEAF_SHARD_FIELDS` at the same indent, so it is anchored with the comment
# line above it. Pass 1 of this battery recorded the ambiguous version as NOT
# APPLIED -- which is the whole reason that status exists as a separate verdict.
ENCODE_SKIP_WITNESS_ENTRY = (
    '    # this is the "required" half, and the range check below is the other.\n'
    '    "rollout_encode_skipped",\n'
)
ENCODE_SKIP_WITNESS_ENTRY_DROPPED = (
    '    # this is the "required" half, and the range check below is the other.\n'
)
ENCODE_SKIP_SHARD_LINE = (
    "            payload[\"rollout_encode_skipped\"] = self.rollout_encode_skipped\n"
)
ENCODE_SKIP_DECISION_LINE = (
    '            "rollout_encode_skipped": int(ledger["rollout_encode_skipped"]),\n'
)
ENCODE_SKIP_RANGE_CHECK = "    if not 0 <= encode_skipped <= leaves_priced:\n"
ROLLOUT_CRATE_REGISTRY_ENTRY = '        "rollout_crate": "_search_rollout_crate",\n'
POST_INIT_MEMBERSHIP = "        if self.leaf_eval not in LEAF_EVAL_SEARCH_METHODS:\n"
SCHEMA_STAMP_LINE = (
    "            payload[\"rollout_leaf_schema\"] = ROLLOUT_LEAF_SHARD_SCHEMA\n"
)
SHARD_WORLDS_LINE = (
    "            payload[\"rollout_leaf_worlds\"] = self.rollout_leaf_worlds\n"
)
SHARD_CONDITIONAL = "        if self.rollout_leaf_modes:\n"
SHARD_FALLBACK_NUMERATOR = (
    "                payload[\"rollout_fallback_fraction\"] = (\n"
    "                    self.rollout_cap_hits + self.rollout_dead_ends\n"
    "                ) / self.rollouts_run\n"
)
V1_SINGLE_BRANCH = "    if ROLLOUT_LEAF_V1_WORLD_FIELD in shard:\n"
UNSTAMPED_BRANCH = "    if version is None:\n"
MIGRATION_COLLAPSE_BRANCH = "    if collapsed != 0:\n"
MIGRATION_DEAD_BRANCH = "    if dead != 0:\n"
ABSORB_GATE = "            if rollout_leaf_eval:\n"
WORLDS_WEIGHT_LINE = "                self.stats.rollout_leaf_worlds += weight\n"
# Present in BOTH `require_rollout_leaf_witness` and
# `require_rollout_leaf_shard_schema`; disambiguated by the line above it, which
# names the mapping each reads from.
PARTITION_CHECK = (
    '    dead = int(witness["rollout_dead_ends"])\n'
    "    if terminal + cap + dead != rollouts_run:\n"
)
PARTITION_CHECK_OFF = (
    '    dead = int(witness["rollout_dead_ends"])\n'
    "    if False:\n"
)
SHARD_PARTITION_CHECK = (
    '    dead = int(shard["rollout_dead_ends"])\n'
    "    if terminal + cap + dead != rollouts_run:\n"
)
SHARD_PARTITION_CHECK_OFF = (
    '    dead = int(shard["rollout_dead_ends"])\n'
    "    if False:\n"
)
QUOTIENT_CHECK = "    if abs(fallback - expected) > 1e-9:\n"
DEGENERATE_CHECK = "    if rollouts_run <= 0 or leaves_priced <= 0:\n"
EMPTY_STATS_BRANCH = "        if not isinstance(stats, Mapping) or not stats:\n"
MISSING_STATS_BRANCH = '        if "policy_stats" not in block:\n'
ARM_CLAIM_CROSSCHECK = '        if "rollout_leaf" in block:\n'
ARM_CLAIMED_NOT_ENGAGED = "            if claimed and not engaged:\n"
ARM_ENGAGED_NOT_CLAIMED = "            if engaged and not claimed:\n"
RECURSE_INTO_VALUES = (
    "        for value in payload.values():\n"
    "            found.extend(_engine_mcts_blocks(value))\n"
)
NO_POINT_RATIO_ANCHOR = "    # NO POINT RATIO, because the two measurements of it disagree"
# The `_search_model` site is the only textually UNIQUE one: it accumulates
# `+= weight` (per world, weighted by the collapse multiplicity) while the two
# sequential paths share `+= 1` verbatim. Anchoring on the shared text matched
# twice and recorded NOT APPLIED in pass 1.
DEPTH_ACCUMULATION_MODEL_PATH = (
    "                self.stats.depth_reached_histogram[reached] += weight\n"
)

#: (name, family, [(target file, old, new), ...]).
#:
#: FAMILIES are load-bearing: the recorded run is asserted to contain every one of
#: them, so a battery cannot quietly become a battery about one finding.
MUTANTS: list[tuple[str, str, list[tuple[Path, str, str]]]] = [
    # ---- A1: the last hop, and the frames around it -------------------------
    (
        "a1_policy_stats_to_none",
        "A1 last hop",
        [(BRIDGE, POLICY_STATS_LINE, '                "policy_stats": None,\n')],
    ),
    (
        "a1_engine_policy_stats_returns_empty",
        "A1 last hop",
        [(BRIDGE, ENGINE_STATS_RETURN, "    return {}\n")],
    ),
    (
        "a1_delete_the_policy_stats_line",
        "A1 last hop",
        [(BRIDGE, POLICY_STATS_LINE, "")],
    ),
    (
        "a1_delete_the_boundary_guard_call",
        "A1 last hop",
        [(BRIDGE, BOUNDARY_CALL, "")],
    ),
    (
        "a1_guard_accepts_an_empty_stats_block",
        "A1 last hop",
        [(BRIDGE, EMPTY_STATS_BRANCH, "        if False:\n")],
    ),
    (
        "a1_guard_accepts_a_missing_stats_key",
        "A1 last hop",
        [(BRIDGE, MISSING_STATS_BRANCH, "        if False:\n")],
    ),
    (
        "a1_guard_stops_recursing_into_nested_arms",
        "A1 last hop",
        [(BRIDGE, RECURSE_INTO_VALUES, "")],
    ),
    (
        "a1_guard_skips_the_arm_claim_crosscheck",
        "A1 last hop",
        [(BRIDGE, ARM_CLAIM_CROSSCHECK, "        if False:\n")],
    ),
    (
        "a1_guard_allows_an_arm_claim_with_no_pricer",
        "A1 last hop",
        [(BRIDGE, ARM_CLAIMED_NOT_ENGAGED, "            if False:\n")],
    ),
    (
        "a1_guard_allows_a_raw_row_carrying_arm_telemetry",
        "A1 last hop",
        [(BRIDGE, ARM_ENGAGED_NOT_CLAIMED, "            if False:\n")],
    ),
    # ---- A2: the half-closed encode-skip counter ----------------------------
    (
        "a2_drop_encode_skipped_from_the_report_fields",
        "A2 encode skip",
        [(ENGINE, ENCODE_SKIP_REPORT_ENTRY, "")],
    ),
    (
        "a2_drop_encode_skipped_from_the_witness_fields",
        "A2 encode skip",
        [(ENGINE, ENCODE_SKIP_WITNESS_ENTRY, ENCODE_SKIP_WITNESS_ENTRY_DROPPED)],
    ),
    (
        "a2_zero_the_shard_column",
        "A2 encode skip",
        [
            (
                ENGINE,
                ENCODE_SKIP_SHARD_LINE,
                '            payload["rollout_encode_skipped"] = 0\n',
            )
        ],
    ),
    (
        "a2_zero_the_per_decision_value",
        "A2 encode skip",
        [(ENGINE, ENCODE_SKIP_DECISION_LINE, '            "rollout_encode_skipped": 0,\n')],
    ),
    (
        "a2_delete_the_range_check",
        "A2 encode skip",
        [(ENGINE, ENCODE_SKIP_RANGE_CHECK, "    if False:\n")],
    ),
    # ---- A4: the instrumentation registry -----------------------------------
    (
        "a4_unregister_a_leaf_eval_path",
        "A4 instrumentation",
        [(ENGINE, ROLLOUT_CRATE_REGISTRY_ENTRY, "")],
    ),
    (
        "a4_post_init_ignores_the_registry",
        "A4 instrumentation",
        [(ENGINE, POST_INIT_MEMBERSHIP, "        if False:\n")],
    ),
    (
        "a4_delete_a_depth_accumulation_site",
        "A4 instrumentation",
        [(ENGINE, DEPTH_ACCUMULATION_MODEL_PATH, "")],
    ),
    (
        "a4_duplicate_a_depth_accumulation_site",
        "A4 instrumentation",
        [
            (
                ENGINE,
                DEPTH_ACCUMULATION_MODEL_PATH,
                DEPTH_ACCUMULATION_MODEL_PATH * 2,
            )
        ],
    ),
    # ---- A5: the retraction guard's third shape ------------------------------
    (
        "a5_quote_a_third_unlisted_ratio",
        "A5 retraction",
        [
            (
                ENGINE,
                NO_POINT_RATIO_ANCHOR,
                "    # Measured: the arm costs 2.7x the raw arm on one core.\n"
                + NO_POINT_RATIO_ANCHOR,
            )
        ],
    ),
    (
        "a5_quote_an_enumerated_ratio_outside_the_paragraph",
        "A5 retraction",
        [
            (
                ENGINE,
                NO_POINT_RATIO_ANCHOR,
                "    # The arm costs 1.8x at R=8 against the raw arm.\n"
                + NO_POINT_RATIO_ANCHOR,
            )
        ],
    ),
    (
        "a5_quote_the_range_derivation_as_fact",
        "A5 retraction",
        [
            (
                ENGINE,
                NO_POINT_RATIO_ANCHOR,
                "    # The arm/raw wall ratio is 1.25-3.98x.\n" + NO_POINT_RATIO_ANCHOR,
            )
        ],
    ),
    # ---- A6: the shard schema -------------------------------------------------
    (
        "a6_drop_the_schema_stamp",
        "A6 schema",
        [(ENGINE, SCHEMA_STAMP_LINE, "")],
    ),
    (
        "a6_rename_the_world_count_to_the_v1_spelling",
        "A6 schema",
        [
            (
                ENGINE,
                SHARD_WORLDS_LINE,
                '            payload["rollout_leaf_world_records"] = '
                "self.rollout_leaf_worlds\n",
            )
        ],
    ),
    (
        "a6_reader_accepts_a_v1_shard",
        "A6 schema",
        [(ENGINE, V1_SINGLE_BRANCH, "    if False:\n")],
    ),
    (
        "a6_reader_accepts_an_unstamped_shard",
        "A6 schema",
        [(ENGINE, UNSTAMPED_BRANCH, "    if False:\n")],
    ),
    (
        "a6_migration_ignores_a_collapsed_draw",
        "A6 schema",
        [(ENGINE, MIGRATION_COLLAPSE_BRANCH, "    if False:\n")],
    ),
    (
        "a6_migration_ignores_a_nonzero_dead_end_count",
        "A6 schema",
        [(ENGINE, MIGRATION_DEAD_BRANCH, "    if False:\n")],
    ),
    (
        "a6_shard_fallback_numerator_drops_dead_ends",
        "A6 schema",
        [
            (
                ENGINE,
                SHARD_FALLBACK_NUMERATOR,
                '                payload["rollout_fallback_fraction"] = (\n'
                "                    self.rollout_cap_hits\n"
                "                ) / self.rollouts_run\n",
            )
        ],
    ),
    (
        "a6_shard_block_becomes_unconditional",
        "A6 schema",
        [(ENGINE, SHARD_CONDITIONAL, "        if True:\n")],
    ),
    (
        "a6_reader_skips_the_shard_partition_check",
        "A6 schema",
        [(ENGINE, SHARD_PARTITION_CHECK, SHARD_PARTITION_CHECK_OFF)],
    ),
    (
        "a6_world_count_accumulates_one_not_weight",
        "A6 schema",
        [
            (
                ENGINE,
                WORLDS_WEIGHT_LINE,
                "                self.stats.rollout_leaf_worlds += 1\n",
            )
        ],
    ),
    # ---- The round-1/round-2 surface, re-covered so the battery is a regression
    # ---- gate rather than a one-revision snapshot ------------------------------
    (
        "prior_delete_the_witness_guard_call",
        "prior rounds",
        [(ENGINE, WITNESS_CALL, "")],
    ),
    (
        "prior_witness_builder_returns_none",
        "prior rounds",
        [
            (
                ENGINE,
                "        rollouts_run = int(ledger[\"rollouts_run\"])\n",
                "        return None\n"
                "        rollouts_run = int(ledger[\"rollouts_run\"])\n",
            )
        ],
    ),
    (
        "prior_absorb_block_never_runs",
        "prior rounds",
        [(ENGINE, ABSORB_GATE, "            if False:\n")],
    ),
    (
        "prior_delete_the_partition_check",
        "prior rounds",
        [(ENGINE, PARTITION_CHECK, PARTITION_CHECK_OFF)],
    ),
    (
        "prior_delete_the_quotient_check",
        "prior rounds",
        [(ENGINE, QUOTIENT_CHECK, "    if False:\n")],
    ),
    (
        "prior_delete_the_degenerate_ledger_check",
        "prior rounds",
        [(ENGINE, DEGENERATE_CHECK, "    if False:\n")],
    ),
]

#: Mutants whose SURVIVAL is expected, each with a written equivalence argument.
#: Empty is not a boast: if one appears, its argument goes here and the gate test
#: requires the argument to be substantive rather than a shrug.
EXPECTED_EQUIVALENT: dict[str, str] = {}

#: The controls, and the verdict each MUST produce. These are what make the counts
#: above meaningful; without them "35 killed" is a claim about a runner nobody
#: checked. `_control_hang` is deliberately slow -- it is the only way to observe
#: that a timeout is not silently a kill.
CONTROLS: list[tuple[str, str, list[tuple[Path, str, str]]]] = [
    ("_control_null", "SURVIVED", []),
    (
        "_control_positive",
        "KILLED",
        [(ENGINE, ENCODE_SKIP_DECISION_LINE, '            "rollout_encode_skipped": 0,\n')],
    ),
    (
        "_control_syntax_error",
        "DID NOT RUN",
        [(ENGINE, "def require_rollout_leaf_witness(\n", "def require_rollout_leaf_witness(((\n")],
    ),
    ("_control_deleted_killer", "DID NOT RUN", []),
    (
        "_control_hang",
        "DID NOT RUN",
        [
            (
                ENGINE,
                "ROLLOUT_LEAF_SHARD_SCHEMA = 2\n",
                "ROLLOUT_LEAF_SHARD_SCHEMA = 2\n"
                "import time as _hang_time; _hang_time.sleep(9000)\n",
            )
        ],
    ),
]

_SUMMARY = re.compile(
    r"(?:(?P<failed>\d+) failed)|(?:(?P<errors>\d+) error)|(?:(?P<passed>\d+) passed)"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _classify(
    completed: "subprocess.CompletedProcess[str] | None",
    stdout: str,
    stderr: str,
    *,
    imported: bool,
) -> tuple[str, str, list[str]]:
    """(status, detail, killers) from a killer run.

    THE ORDER OF THESE BRANCHES IS THE WHOLE POINT. `pytest` exits 1 on a collection
    error, so a return-code-only classifier reports an unimportable tree as a clean
    sweep of kills. Every non-failure exit is therefore DID NOT RUN.

    `imported` is passed in rather than sniffed out of the output: the first version
    of this looked for the marker string in the combined blob, and Python echoes the
    failing `-c` SOURCE LINE in its traceback -- which contains the marker -- so an
    unimportable tree read as "imported fine". A control caught it. That is the same
    class of self-reference the battery exists to find, occurring in the battery.
    """

    if completed is None:
        return "DID NOT RUN", "timeout: the killer run did not terminate", []
    if not imported:
        return (
            "DID NOT RUN",
            "the mutated tree does not import; no test was ever collected",
            [],
        )
    blob = stdout + stderr
    failed = errors = passed = 0
    for match in _SUMMARY.finditer(blob):
        if match.group("failed"):
            failed = max(failed, int(match.group("failed")))
        if match.group("errors"):
            errors = max(errors, int(match.group("errors")))
        if match.group("passed"):
            passed = max(passed, int(match.group("passed")))
    named = sorted(
        {
            line.split()[1]
            for line in blob.splitlines()
            if line.startswith("FAILED ") and len(line.split()) > 1
        }
    )
    if errors:
        return (
            "DID NOT RUN",
            f"pytest reported {errors} collection/setup error(s); no test was run",
            [],
        )
    if completed.returncode == 0:
        if passed == 0:
            return "DID NOT RUN", "a green run that collected nothing", []
        return "SURVIVED", f"{passed} passed", []
    if failed == 0:
        return (
            "DID NOT RUN",
            f"exit {completed.returncode} with zero reported failures",
            [],
        )
    if not named:
        return (
            "DID NOT RUN",
            f"{failed} failure(s) reported but no FAILED nodeid named",
            [],
        )
    return "KILLED", f"{failed} failed, {passed} passed", named


def _run_killers(python: str, timeout: int) -> tuple[str, str, str, object, bool]:
    """Resolve the modules, then run the killers.

    Returns (stdout, stderr, resolved-marker-lines, completed, imported).
    """

    env = dict(os.environ)
    # DELIBERATELY POISONED: a sibling checkout on PYTHONPATH is the exact condition
    # under which a mutant appears to survive without loading. `tests/conftest.py`
    # moves this tree's `src` to the FRONT, so the resolved path printed below must
    # still be this tree's -- and the classifier refuses the run if it is not.
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # ONE PROCESS, and a real script file rather than `-c`. Two reasons, in order:
    # importing `foulplay_bridge` drags in torch, so a separate probe process doubled
    # the wall clock of every mutant; and Python echoes the failing source line of a
    # `-c` program in its traceback, which put the resolved-path MARKER into the
    # output of a run that never imported anything. A file's traceback quotes the
    # file, and the marker is only ever emitted by a successful `print`.
    probe_script = ROOT / ".mutate_probe.py"
    probe_script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        f"sys.path.insert(0, {str(ROOT / 'tests')!r})\n"
        "import pokezero.engine_search as _es, pokezero.foulplay_bridge as _fb\n"
        "print('RESOLVED_ENGINE=' + _es.__file__)\n"
        "print('RESOLVED_BRIDGE=' + _fb.__file__)\n"
        "sys.stdout.flush()\n"
        "import pytest\n"
        f"raise SystemExit(pytest.main({[*KILLERS, '-q', '--tb=no', '-p', 'no:cacheprovider']!r}))\n"
    )
    try:
        completed = subprocess.run(
            [python, str(probe_script)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        probe_script.unlink(missing_ok=True)
        return "", "", "", None, False
    finally:
        probe_script.unlink(missing_ok=True)
    resolved = "\n".join(
        line for line in completed.stdout.splitlines() if line.startswith("RESOLVED_")
    )
    imported = "RESOLVED_ENGINE=" in resolved
    return completed.stdout, completed.stderr, resolved, completed, imported


def _apply(edits: list[tuple[Path, str, str]], originals: dict[Path, str]) -> str | None:
    """Write the mutant. Returns a reason string when it cannot be applied."""

    staged: dict[Path, str] = {}
    for path, old, new in edits:
        text = staged.get(path, originals[path])
        count = text.count(old)
        if count != 1:
            return f"anchor matched {count} times in {path.name}"
        staged[path] = text.replace(old, new)
    for path, text in staged.items():
        path.write_text(text)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-python", default=sys.executable)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--only", default=None, help="substring filter over mutant names (debugging)"
    )
    parser.add_argument(
        "--check-anchors",
        action="store_true",
        help=(
            "verify every anchor matches exactly once and exit, without running the "
            "killers. Pass 1 of this battery burned 40 minutes to discover four "
            "ambiguous anchors, which is a fact about the harness rather than about "
            "the tree and does not need a test run to establish."
        ),
    )
    args = parser.parse_args(argv)

    if args.check_anchors:
        bad: list[str] = []
        for name, _family, edits in MUTANTS + CONTROLS:
            for path, old, _new in edits:
                count = path.read_text().count(old)
                if count != 1:
                    bad.append(f"{name}: {path.name} anchor matched {count} times")
        for line in bad:
            print(line, file=sys.stderr)
        print(json.dumps({"ambiguous_or_missing_anchors": len(bad)}))
        return 1 if bad else 0

    targets = sorted({path for _, _, edits in MUTANTS + CONTROLS for path, _, _ in edits})
    originals = {path: path.read_text() for path in targets}
    before = {path: _sha256(path) for path in targets}
    killer_paths = [ROOT / name for name in KILLERS]
    killer_originals = {path: path.read_text() for path in killer_paths}

    results: list[dict[str, object]] = []
    control_results: list[dict[str, object]] = []
    resolved_seen: set[str] = set()

    try:
        for table, sink, is_control in ((MUTANTS, results, False), (CONTROLS, control_results, True)):
            for entry in table:
                name = entry[0]
                edits = entry[-1]
                if args.only and args.only not in name:
                    continue
                for path, text in originals.items():
                    path.write_text(text)
                for path, text in killer_originals.items():
                    path.write_text(text)
                reason = _apply(edits, originals)
                if reason is not None:
                    sink.append(
                        {
                            "name": name,
                            "family": entry[1] if not is_control else "control",
                            "status": "NOT APPLIED",
                            "detail": reason,
                            "killers": [],
                        }
                    )
                    continue
                if name == "_control_deleted_killer":
                    (ROOT / KILLERS[0]).unlink()
                stdout, stderr, resolved, completed, imported = _run_killers(
                    args.venv_python, args.timeout
                )
                status, detail, killers = _classify(
                    completed, stdout, stderr, imported=imported
                )
                for line in resolved.splitlines():
                    if line.startswith("RESOLVED_"):
                        resolved_seen.add(line.strip())
                record = {
                    "name": name,
                    "family": "control" if is_control else entry[1],
                    "status": status,
                    "detail": detail,
                    "killers": killers,
                }
                if is_control:
                    record["required"] = entry[1]
                sink.append(record)
                print(
                    f"{status:12s} {name}  ({detail})",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for path, text in originals.items():
            path.write_text(text)
        for path, text in killer_originals.items():
            path.write_text(text)

    after = {path: _sha256(path) for path in targets}
    if before != after:
        raise InstrumentFailure(
            "the tree was not restored byte-for-byte: "
            f"{[p.name for p in targets if before[p] != after[p]]}"
        )
    for path in killer_paths:
        if path.read_text() != killer_originals[path]:
            raise InstrumentFailure(f"killer module not restored: {path.name}")

    # The resolved-path proof: every run must have imported THIS tree, despite a
    # PYTHONPATH that points at it by a different route.
    expected = {
        f"RESOLVED_ENGINE={ENGINE}",
        f"RESOLVED_BRIDGE={BRIDGE}",
    }
    unexpected = sorted(resolved_seen - expected)
    if unexpected:
        raise InstrumentFailure(
            f"a run imported a module outside this tree: {unexpected}"
        )
    if not args.only and not expected <= resolved_seen:
        raise InstrumentFailure(
            "no run ever reported a resolved module path; the battery cannot say "
            "which copy of the code it mutated"
        )

    # The controls decide whether the counts mean anything.
    control_failures = [
        c for c in control_results if c["status"] != c.get("required")
    ]
    doc = {
        "_README": (
            "Recorded by scripts/mutate_rollout_leaf_witness.py. NEVER edit by hand "
            "to make the gate pass -- tests/test_rollout_leaf_witness_mutation_"
            "battery.py holds this file to the harness that produced it, including "
            "the harness's own sha256."
        ),
        "harness_sha256": _sha256(Path(__file__).resolve()),
        "targets": {path.name: before[path] for path in targets},
        "killers": list(KILLERS),
        "applied": sum(1 for r in results if r["status"] != "NOT APPLIED"),
        "killed": sum(1 for r in results if r["status"] == "KILLED"),
        "survived": sum(1 for r in results if r["status"] == "SURVIVED"),
        "did_not_run": sum(1 for r in results if r["status"] == "DID NOT RUN"),
        "not_applied": sum(1 for r in results if r["status"] == "NOT APPLIED"),
        "expected_equivalent": dict(EXPECTED_EQUIVALENT),
        "mutants": results,
        "controls": control_results,
        "resolved_modules": sorted(resolved_seen),
    }
    if args.json and not args.only:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json}", file=sys.stderr)
    print(
        json.dumps(
            {k: doc[k] for k in ("applied", "killed", "survived", "did_not_run", "not_applied")},
            sort_keys=True,
        )
    )
    if control_failures:
        raise InstrumentFailure(
            "controls did not produce their required verdicts: "
            + ", ".join(
                f"{c['name']} -> {c['status']} (required {c['required']})"
                for c in control_failures
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
