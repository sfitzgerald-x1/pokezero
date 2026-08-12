#!/usr/bin/env python3
"""C157 mutation battery over every conjunct of `volatile_fail` in `events.rs`.

WHY THIS IS COMMITTED. A claim about an instrument nobody else can run is not evidence, and
review could not reproduce this battery's self-test because the harness lived in `/tmp`. It is
the same rule the campaign already applies to counts: publish the SCRIPT, never a description
of it.

WHAT IT FIXES ABOUT ITS OWN FIRST VERSION. The first harness anchored each mutant on a single
line of source. Three of those anchors were not unique in the file, the string replacement
asserted and wrote nothing, and the loop went on to report `SURVIVED` for a mutant that had
never been APPLIED -- the "tally that cannot express a survivor" defect, committed inside the
check for it. So:

  * every mutant is anchored INSIDE the predicate block, which is located whole and required
    to occur exactly once;
  * a mutant whose anchor is missing, ambiguous, or which produces no textual change is an
    INSTRUMENT FAILURE with a non-zero exit, never a verdict;
  * `--self-test` proves that, by asking for a mutant whose anchor cannot exist and requiring
    the run to fail;
  * a run reporting fewer than `--min-binaries` test binaries is an INSTRUMENT FAILURE too,
    because a build break otherwise reads as "nothing failed". Test binaries also SIGABRT
    without libtorch while emitting no `test result: FAILED` line, so the binary COUNT is the
    discriminator and not the failure count.

⚠ `--min-binaries` DEFAULTED TO 30 AND THAT WAS BELOW THE FAILURE MODE IT EXISTS TO CATCH.
Measured: a `--features model` run with no `DYLD_LIBRARY_PATH` reports **32** binaries, 246
passed, 0 failed, with 4 binaries SIGABRT-ing silently. A threshold of 30 therefore accepted
the exact condition the guard was written for. It is 34 now, above that 32 and below the 36
this crate reports when whole.

Usage:
  scripts/c157_volatile_fail_conjunct_battery.py --self-test
  scripts/c157_volatile_fail_conjunct_battery.py [--only NAME ...]

Leaves the tree byte-identical to how it found it, and verifies that on the way out.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVENTS = REPO / "rust" / "pokezero-search" / "src" / "events.rs"
CRATE = REPO / "rust" / "pokezero-search"

#: The predicate, located whole so no mutant anchor can be ambiguous.
BLOCK = """    let volatile_fail = !has_any_effect
        && !leechseed_grass_immune
        && choice.category == MoveCategory::Status
        && choice.status.is_none()
        && choice.side_condition.is_none()
        && choice.target == MoveTarget::Opponent
        && !non_ghost_curse
        && choice.accuracy < 100.0
        && effectiveness > 0.0
        && no_effect_hit_outweighs_miss(choice.accuracy)
        && choice.volatile_status.as_ref().map_or(false, |vs| {
            vs.target == MoveTarget::Opponent && {"""

#: The `|-immune|` predicate and its render site. A SECOND block, because the battery covered
#: only `volatile_fail` and review found three surviving stricter mutants on this one -- the
#: pure-Grass narrowing alone silently reverted 122 of the 168 Grass variants, including the
#: Grass/Dark species whose captured line is this change's cited evidence.
GRASS_BLOCK = """    let leechseed_grass_immune = !has_any_effect
        && choice.move_id == Choices::LEECHSEED
        && {
            let d = match defender {
                SideReference::SideOne => &sim.state.side_one,
                SideReference::SideTwo => &sim.state.side_two,
            };
            d.get_active_immutable().has_type(&PokemonType::GRASS)
        };"""

RENDER_BLOCK = """        if leechseed_grass_immune && !defender_protected {"""

GRASS_MUTANTS: dict[str, tuple[str, str, str]] = {
    # name: (block, old, new)
    "stricter_pure_grass_only": (
        GRASS_BLOCK,
        "            d.get_active_immutable().has_type(&PokemonType::GRASS)",
        "            d.get_active_immutable().types == (PokemonType::GRASS, PokemonType::TYPELESS) // MUTANT",
    ),
    "drop_move_id_scope": (
        GRASS_BLOCK,
        "        && choice.move_id == Choices::LEECHSEED\n",
        "        && true // MUTANT\n",
    ),
    "drop_has_any_effect_grass": (
        GRASS_BLOCK,
        "    let leechseed_grass_immune = !has_any_effect\n",
        "    let leechseed_grass_immune = true // MUTANT\n",
    ),
    "drop_render_protect_guard": (
        RENDER_BLOCK,
        "        if leechseed_grass_immune && !defender_protected {",
        "        if leechseed_grass_immune { // MUTANT",
    ),
}

MUTANTS: dict[str, tuple[str, str]] = {
    "drop_grass_immune":        ("        && !leechseed_grass_immune\n", "        && true // MUTANT\n"),
    "drop_category_status":     ("        && choice.category == MoveCategory::Status\n", "        && true // MUTANT\n"),
    "drop_status_is_none":      ("        && choice.status.is_none()\n", "        && true // MUTANT\n"),
    "drop_side_condition":      ("        && choice.side_condition.is_none()\n", "        && true // MUTANT\n"),
    "drop_choice_target_opp":   ("        && choice.target == MoveTarget::Opponent\n", "        && true // MUTANT\n"),
    "drop_non_ghost_curse":     ("        && !non_ghost_curse\n", "        && true // MUTANT\n"),
    "drop_accuracy_lt_100":     ("        && choice.accuracy < 100.0\n", "        && true // MUTANT\n"),
    "drop_effectiveness":       ("        && effectiveness > 0.0\n", "        && true // MUTANT\n"),
    "drop_dominance":           ("        && no_effect_hit_outweighs_miss(choice.accuracy)\n", "        && true // MUTANT\n"),
    "stricter_acc_ge_90":       ("        && no_effect_hit_outweighs_miss(choice.accuracy)\n", "        && choice.accuracy >= 90.0 // MUTANT\n"),
    "drop_vs_target_opp":       ("            vs.target == MoveTarget::Opponent && {", "            true && { // MUTANT"),
    # The three conjuncts DELETED in review round 2. Re-adding a dead conjunct must be
    # behaviour-neutral; re-adding the live one must be caught.
    "readd_absorb":             ("        && effectiveness > 0.0\n", "        && effectiveness > 0.0\n        && absorb.is_none() // MUTANT\n"),
    "readd_defender_protected": ("        && effectiveness > 0.0\n", "        && effectiveness > 0.0\n        && !defender_protected // MUTANT\n"),
    "readd_ability_immune":     ("        && effectiveness > 0.0\n", "        && effectiveness > 0.0\n        && ability_immune.is_none() // MUTANT\n"),
    # Anchor that cannot exist, for --self-test.
    "_bogus_self_test":         ("        && this_conjunct_does_not_exist\n", "        && true\n"),
}


class InstrumentFailure(RuntimeError):
    """Never a verdict. Always a non-zero exit."""


def _read() -> str:
    return EVENTS.read_text(encoding="utf-8")


def apply(name: str) -> str:
    if name in GRASS_MUTANTS:
        return _apply_in(name, *GRASS_MUTANTS[name])
    old, new = MUTANTS[name]
    return _apply_in(name, BLOCK, old, new)


def _apply_in(name: str, block: str, old: str, new: str) -> str:
    source = _read()
    if source.count(block) != 1:
        raise InstrumentFailure(
            f"the anchor block for {name} was found {source.count(block)} times, not once; "
            "the harness is anchored to source that has moved"
        )
    if block.count(old) != 1:
        raise InstrumentFailure(
            f"{name}: anchor occurs {block.count(old)} times INSIDE its block, not once"
        )
    mutated = source.replace(block, block.replace(old, new), 1)
    if mutated == source:
        raise InstrumentFailure(f"{name}: produced no textual change")
    EVENTS.write_text(mutated, encoding="utf-8")
    return source


def run_suite(min_binaries: int) -> tuple[int, int, int]:
    completed = subprocess.run(
        ["cargo", "test", "--no-fail-fast"], cwd=CRATE, capture_output=True, text=True
    )
    lines = [l for l in completed.stdout.splitlines() if l.startswith("test result")]
    if len(lines) < min_binaries:
        raise InstrumentFailure(
            f"only {len(lines)} test binaries reported (expected >= {min_binaries}). A build "
            "break or a SIGABRT emits no `test result: FAILED`, so this is not a survivor."
        )
    failed = sum(1 for l in lines if "FAILED" in l)
    passed = sum(int(l.split("ok. ")[1].split(" passed")[0]) for l in lines if "ok. " in l)
    return len(lines), passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--min-binaries", type=int, default=34)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        try:
            apply("_bogus_self_test")
        except InstrumentFailure as error:
            print(f"SELF-TEST PASS: a bogus anchor is an INSTRUMENT FAILURE, not a survivor\n  {error}")
            return 0
        EVENTS.write_text(_read(), encoding="utf-8")
        print("SELF-TEST FAIL: the harness accepted a mutant whose anchor cannot exist")
        return 1

    before = _read()
    names = args.only or (
        [n for n in MUTANTS if not n.startswith("_")] + list(GRASS_MUTANTS)
    )
    worst = 0
    for name in names:
        try:
            original = apply(name)
        except InstrumentFailure as error:
            print(f"{name:26} INSTRUMENT FAILURE: {error}")
            worst = 1
            continue
        try:
            binaries, passed, failed = run_suite(args.min_binaries)
            verdict = (
                f"KILLED   ({failed} of {binaries} binaries)"
                if failed
                else f"SURVIVED ({binaries} binaries, {passed} passed)"
            )
        except InstrumentFailure as error:
            verdict = f"INSTRUMENT FAILURE: {error}"
            worst = 1
        finally:
            EVENTS.write_text(original, encoding="utf-8")
        print(f"{name:26} {verdict}")

    if _read() != before:
        print("INSTRUMENT FAILURE: the tree was not restored", file=sys.stderr)
        return 1
    print("tree restored byte-identically")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
