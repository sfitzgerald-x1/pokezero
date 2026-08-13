#!/usr/bin/env python3
"""Mutation battery for the Sleep Talk zero-heal discriminator (`events.rs`).

WHY THIS IS COMMITTED. It used to live outside the tree and existed only as commit prose,
which review flagged: a battery nobody can re-run is a claim, not a measurement. It writes a
machine-readable result, and `tests/test_zero_heal_guard_mutation_battery.py` holds the
recorded result to it.

BOTH DIRECTIONS, which is the point. Fail-open mutants (the guard stops refusing something it
must) AND fail-safe mutants -- strictly MORE conservative variants. A surviving conservative
mutant means the suite is silent exactly where an over-refusal lives, and reverting the fix
cannot show that.

DEATH CAUSES ARE AUDITED. Every mutant is compiled before it is scored, so a kill is a failed
assertion and not a build error, and the crate suite runs with `--no-fail-fast`. Without that
flag cargo stops after the first failing target: measured on this tree, 4 of 36 binaries ran
versus 36 with it, so three fail-open mutants were first recorded as killed by a UNIT test of
the predicate while the end-to-end fixtures never executed. That was a death-cause artifact,
not a kill.

  usage: scripts/mutate_zero_heal_guard.py --venv-python .venv/bin/python \
             --json reports/artifacts/zero_heal_guard_mutation_battery.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "rust/pokezero-search/src/events.rs"
CRATE = REPO / "rust/pokezero-search"
PY_MODULE = "tests.test_crate_protect_marker_state_reads"

#: The conjunct this whole change adds, at the production read site.
CONJUNCT = "clamps && callee_can_convert_an_opponent_heal,"
#: The discriminator's body -- producer 2's own `if`.
PRED_BODY = """    matches!(
        choice.heal,
        Some(poke_engine::choices::Heal {
            target: MoveTarget::Opponent,
            amount
        }) if amount > 0.0
    )"""

MUTANTS: list[tuple[str, str, list[tuple[str, str]]]] = [
    # ---- FAIL-OPEN: renders where it must refuse --------------------------------------
    ("open_predicate_always_false", "fail-open",
     [(PRED_BODY, "    let _ = choice;\n    false")]),
    ("open_target_user_not_opponent", "fail-open",
     [("target: MoveTarget::Opponent,\n            amount\n        }) if amount > 0.0",
       "target: MoveTarget::User,\n            amount\n        }) if amount > 0.0")]),
    ("open_amount_gt_one", "fail-open", [("}) if amount > 0.0", "}) if amount > 1.0")]),
    ("open_early_return_on_second_match", "fail-open",
     [("            match_count += 1;\n            if matched.is_none() {\n"
       "                matched = Some(choice);\n            }",
       "            match_count += 1;\n            if matched.is_some() {\n"
       "                return SleepTalkProbe { ident: SleepTalkIdent::Ambiguous,\n"
       "                    callee_can_convert_an_opponent_heal: can_convert_an_opponent_heal };\n"
       "            }\n            matched = Some(choice);")]),
    ("open_scan_only_matching_candidates", "fail-open",
     [("        can_convert_an_opponent_heal |= choice_can_convert_an_opponent_heal(&choice);\n"
       "        if generated", "        if generated"),
      ("            match_count += 1;\n            if matched.is_none() {",
       "            can_convert_an_opponent_heal |= "
       "choice_can_convert_an_opponent_heal(&choice);\n"
       "            match_count += 1;\n            if matched.is_none() {")]),
    ("open_flag_true_on_empty_candidate_list", "fail-safe-shaped, listed with the open set "
                                              "because review proposed it",
     [("    let mut can_convert_an_opponent_heal = false;",
       "    let mut can_convert_an_opponent_heal = candidates.is_empty();")]),
    # ---- FAIL-SAFE: refuses where it may render (must ALSO die) -----------------------
    ("safe_revert_the_conjunct", "fail-safe (= pre-PR)", [(CONJUNCT, "clamps,")]),
    ("safe_predicate_always_true", "fail-safe",
     [(PRED_BODY, "    let _ = choice;\n    true")]),
    ("safe_ability_presence_only", "fail-safe (= pre-#1211)", [(CONJUNCT, "has_absorb,")]),
    ("safe_hardcode_absorb_possible", "fail-safe", [(CONJUNCT, "true,")]),
    ("safe_amount_ge_zero", "fail-safe", [("}) if amount > 0.0", "}) if amount >= 0.0")]),
    # ---- COUNTER: stops refusing without starting to count ---------------------------
    ("counter_collapsed_into_headroom", "counter",
     [("        (true, true) => PROTECT_MARKER_RENDERED_ABSORB_FULL_HP,",
       "        (true, true) => PROTECT_MARKER_RENDERED_ABSORB_HEADROOM,")]),
    ("counter_deleted_for_the_new_arm", "counter",
     [('    "protect_marker_rendered_absorb_full_hp",\n', "")]),
    # ---- REFACTOR sanity -------------------------------------------------------------
    ("refactor_match_count_gt_zero", "refactor",
     [("Some(_) if match_count > 1 =>", "Some(_) if match_count > 0 =>")]),
]

#: Mutants that survive because they are EQUIVALENT, with the reason. A survivor absent from
#: here is a suite gap; an entry here that gets killed is a stale justification. Both red.
EXPECTED_EQUIVALENT: dict[str, str] = {
    "open_scan_only_matching_candidates":
        "Equivalent on the Ambiguous arm: if producer 2 fired, the callee that fired is the "
        "one that generated this tail, so it matches by construction and a matching-only "
        "scan cannot miss it. Scoped in SleepTalkProbe's doc, which also records that the "
        "argument is FALSE as stated on NoneMatched and what bounds it there.",
    "open_flag_true_on_empty_candidate_list":
        "Unreachable arm. Measured against a control: with the sleeper's only move being "
        "Sleep Talk the engine emits one branch, the 50% 'nothing happened' arm, and no "
        "branch carrying a callee tail -- so no branch exists on which the flag is "
        "consulted. The two-callee control emits the second branch and does refuse.",
}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venv-python", default=str(REPO / ".venv/bin/python"))
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--logs", type=pathlib.Path, default=pathlib.Path("/tmp/zhg-mutants"))
    ap.add_argument("--skip-python", action="store_true",
                    help="crate suite only; leaves the venv wheel untouched")
    args = ap.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    original = SRC.read_text()
    results = []
    try:
        for name, direction, edits in MUTANTS:
            SRC.write_text(original)
            text, applied = original, True
            for old, new in edits:
                if text.count(old) != 1:
                    results.append({"name": name, "direction": direction,
                                    "status": "NOT_APPLIED",
                                    "detail": f"anchor matched {text.count(old)} times",
                                    "killers": []})
                    applied = False
                    break
                text = text.replace(old, new)
            if not applied:
                continue
            SRC.write_text(text)
            env = dict(os.environ, RUSTFLAGS="-C debug-assertions=yes")
            build = subprocess.run(
                ["cargo", "test", "--release", "--no-fail-fast", "--no-run"],
                cwd=CRATE, env=env, capture_output=True, text=True)
            if build.returncode:
                (args.logs / f"{name}.build").write_text(build.stdout + build.stderr)
                results.append({"name": name, "direction": direction,
                                "status": "NOT_APPLIED", "detail": "compile error",
                                "killers": []})
                continue
            crate = subprocess.run(
                ["cargo", "test", "--release", "--no-fail-fast"],
                cwd=CRATE, env=env, capture_output=True, text=True)
            (args.logs / f"{name}.cargo").write_text(crate.stdout + crate.stderr)
            killers = sorted({
                line.split()[1] for line in crate.stdout.splitlines()
                if line.startswith("test ") and line.endswith("... FAILED")
            })
            py_rc = 0
            if not args.skip_python:
                subprocess.run(["bash", str(REPO / "scripts/build_search_crate_model.sh")],
                               capture_output=True, text=True)
                pyrun = subprocess.run(
                    [args.venv_python, "-B", "-m", "unittest", PY_MODULE, "-v"],
                    cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO / "src")),
                    capture_output=True, text=True)
                (args.logs / f"{name}.py").write_text(pyrun.stdout + pyrun.stderr)
                py_rc = pyrun.returncode
                killers += sorted({
                    line.split()[1].split("(")[0].strip()
                    for line in (pyrun.stdout + pyrun.stderr).splitlines()
                    if line.startswith("FAIL: ") or line.startswith("ERROR: ")
                })
            dead = bool(crate.returncode) or bool(py_rc)
            results.append({"name": name, "direction": direction,
                            "status": "KILLED" if dead else "SURVIVED",
                            "detail": "", "killers": sorted(set(killers))})
            print(f"{name:44} {direction:24} {results[-1]['status']}", flush=True)
    finally:
        SRC.write_text(original)

    doc = {
        "_README": (
            "Recorded run of scripts/mutate_zero_heal_guard.py. Regenerate with that script; "
            "tests/test_zero_heal_guard_mutation_battery.py holds this to the harness's own "
            "MUTANTS table and to EXPECTED_EQUIVALENT, so a hand edit or a stale re-run is "
            "loud. NEVER edit by hand to make the gate pass."),
        "harness_sha256": _sha256(pathlib.Path(__file__)),
        "events_rs_sha256": _sha256(SRC),
        "applied": sum(1 for r in results if r["status"] != "NOT_APPLIED"),
        "killed": sum(1 for r in results if r["status"] == "KILLED"),
        "survived": sum(1 for r in results if r["status"] == "SURVIVED"),
        "never_applied": sum(1 for r in results if r["status"] == "NOT_APPLIED"),
        "expected_equivalent": EXPECTED_EQUIVALENT,
        "mutants": results,
    }
    out = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(out)
        print(f"wrote {args.json}")
    else:
        print(out, end="")
    print(f"applied {doc['applied']}  killed {doc['killed']}  survived {doc['survived']}"
          f"  never_applied {doc['never_applied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
