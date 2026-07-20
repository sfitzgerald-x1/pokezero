#!/usr/bin/env python
"""Showdown-vs-engine differential for the gen3 Attract-immobilization patch.

This is the residual-order-caliber ground-truth gate for
``third_party/poke-engine-gen3-attract.patch``. It drives two curated gen3
Custom Game scenarios through BOTH:

  * the real Node Showdown sim (``pokezero.showdown_fixture`` -> ``battle_bridge.mjs``),
    over N seeds, counting how often the attracted mon is immobilized vs moves,
    and asserting the exact protocol shapes; and
  * the patched ``poke_engine`` (``generate_instructions``), reading the exact
    move/immobilize branch probabilities.

Scenarios:
  free   : p2 (male) is attracted by p1 (female Attract). Expect ~50% immobilize.
  para   : p2 is Thunder-Wave paralyzed THEN attracted. Expect ~37.5% move
           (0.75 * 0.5), with the immobilization split ~50% Attract / ~12.5% par
           because Showdown resolves Attract (onBeforeMove priority 2) before
           paralysis (priority 1).

The attracted mon uses Curse (a Normal-type stat move: no damage to p1, so
neither side faints and there are no force-switch boundaries) as the observable
"did it move?" probe. p1 uses only status moves, so it survives indefinitely.

MUST run in the dedicated attract venv (never the shared one):
    .venv-attract/bin/python scripts/attract_differential.py \
        --showdown-root /Users/scott/workspace/pokerena/vendor/pokemon-showdown
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokezero.local_showdown import LocalShowdownConfig
from pokezero.showdown_fixture import FixturePokemon, run_multi_turn_fixture

try:
    import poke_engine
except ImportError:  # pragma: no cover
    poke_engine = None


# --- teams -----------------------------------------------------------------

def _clefable() -> FixturePokemon:
    # Female Attract user; only status moves -> never faints, never damages p2.
    return FixturePokemon(
        species="Clefable", gender="F", ability="Natural Cure", item="Leftovers",
        moves=("Attract", "Thunder Wave", "Soft-Boiled", "Seismic Toss"),
    )


def _snorlax() -> FixturePokemon:
    # Male target; Curse is the observable no-damage-to-p1 probe move.
    return FixturePokemon(
        species="Snorlax", gender="M", ability="Thick Fat", item="Leftovers",
        moves=("Curse", "Body Slam", "Rest", "Shadow Ball"),
    )


# --- Showdown ground truth -------------------------------------------------

_P2 = "p2a"


@dataclass
class ShowdownTally:
    seeds: int = 0
    moved: int = 0
    cant_attract: int = 0
    cant_par: int = 0
    activate_seen: int = 0
    other: int = 0

    @property
    def immobilized(self) -> int:
        return self.cant_attract + self.cant_par


def _classify_step(lines: tuple[str, ...]) -> str:
    """Classify p2's measured turn from its protocol lines."""

    saw_move = any(l.startswith(f"|move|{_P2}") and "|Curse" in l for l in lines)
    saw_cant_attract = any(
        l.startswith(f"|cant|{_P2}") and l.rstrip().endswith("|Attract") for l in lines
    )
    saw_cant_par = any(
        l.startswith(f"|cant|{_P2}") and l.rstrip().endswith("|par") for l in lines
    )
    if saw_cant_attract:
        return "cant_attract"
    if saw_cant_par:
        return "cant_par"
    if saw_move:
        return "moved"
    return "other"


def _activate_line_present(lines: tuple[str, ...]) -> bool:
    # |-activate|p2a: Snorlax|move: Attract|[of] p1a: Clefable
    return any(
        l.startswith(f"|-activate|{_P2}") and "|move: Attract" in l and "[of] p1a" in l
        for l in lines
    )


def run_showdown(scenario: str, *, seeds: int, seed_start: int,
                 config: LocalShowdownConfig, verbose: bool) -> tuple[ShowdownTally, list[str]]:
    tally = ShowdownTally()
    sample_lines: list[str] = []
    p1, p2 = _clefable(), _snorlax()
    if scenario == "free":
        turns = [("move attract", "move curse"), ("move softboiled", "move curse")]
        measured = 1  # 0-based index of the measured step (turn 2)
    elif scenario == "para":
        turns = [
            ("move thunderwave", "move curse"),
            ("move attract", "move curse"),
            ("move softboiled", "move curse"),
        ]
        measured = 2  # turn 3
    else:
        raise ValueError(scenario)

    for seed in range(seed_start, seed_start + seeds):
        result = run_multi_turn_fixture(
            p1_team=[p1], p2_team=[p2], turns=turns, seed=seed, config=config,
        )
        if len(result.steps) <= measured:
            tally.other += 1
            continue
        lines = result.steps[measured].protocol_lines
        tally.seeds += 1
        if _activate_line_present(lines):
            tally.activate_seen += 1
        kind = _classify_step(lines)
        setattr(tally, kind, getattr(tally, kind) + 1)
        if verbose and len(sample_lines) < 40 and seed == seed_start:
            sample_lines = [l for l in lines if l.startswith(f"|move|{_P2}")
                            or l.startswith(f"|cant|{_P2}") or f"|{_P2}" in l and "Attract" in l]
    return tally, sample_lines


# --- Engine ground truth ---------------------------------------------------

def engine_probs(*, paralyzed: bool) -> tuple[float, float]:
    """Return (moved%, immobilized%) from the patched engine for an attracted mon.

    Move-agnostic: uses Swords Dance (single deterministic self-boost branch) so
    'moved' <=> a Boost instruction is present. Mirrors the pin-test fixture.
    """

    assert poke_engine is not None
    pe = poke_engine
    status = "paralyze" if paralyzed else "none"
    dummy = pe.Pokemon(id="pikachu", level=1, hp=0)

    def mon(species, moves, **kw):
        return pe.Pokemon(
            id=species, level=80, types=("normal", "typeless"), hp=300, maxhp=300,
            ability="innerfocus", item="none", attack=180, defense=180,
            special_attack=180, special_defense=180, speed=120,
            moves=[pe.Move(id=m, pp=16) for m in moves], **kw,
        )

    side_one = pe.Side(
        active_index="0",
        pokemon=[mon("snorlax", ["swordsdance", "bodyslam"], status=status)] + [dummy] * 5,
        volatile_statuses={"ATTRACT"},
    )
    side_two = pe.Side(active_index="0", pokemon=[mon("wobbuffet", ["splash", "tackle"])] + [dummy] * 5)
    state = pe.State(side_one=side_one, side_two=side_two, weather="none", terrain="none", trick_room=False)
    branches = pe.generate_instructions(state, "swordsdance", "splash")
    moved = sum(b.percentage for b in branches
                if any("Boost SideOne" in str(i) for i in b.instruction_list))
    immob = sum(b.percentage for b in branches
                if not any("Boost SideOne" in str(i) for i in b.instruction_list))
    return moved, immob


# --- driver ----------------------------------------------------------------

def _binom_ok(observed_frac: float, expected: float, n: int, sigmas: float = 4.0) -> bool:
    if n == 0:
        return False
    std = math.sqrt(max(expected * (1 - expected), 1e-9) / n)
    return abs(observed_frac - expected) <= sigmas * std + 1e-9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--showdown-root", required=True)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=770000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if poke_engine is None:
        print("FAIL: poke_engine not importable (build the patched wheel into this venv)")
        return 2

    config = LocalShowdownConfig(showdown_root=args.showdown_root)
    failures: list[str] = []

    # Engine side is exact and move-agnostic.
    eng_free_moved, eng_free_immob = engine_probs(paralyzed=False)
    eng_para_moved, eng_para_immob = engine_probs(paralyzed=True)

    print("=" * 78)
    print("gen3 Attract-immobilization differential  (Showdown ground truth vs patched engine)")
    print("=" * 78)

    for scenario, eng_moved, eng_immob in (
        ("free", eng_free_moved, eng_free_immob),
        ("para", eng_para_moved, eng_para_immob),
    ):
        tally, sample = run_showdown(
            scenario, seeds=args.seeds, seed_start=args.seed_start,
            config=config, verbose=args.verbose,
        )
        n = tally.seeds
        moved_frac = tally.moved / n if n else 0.0
        immob_frac = tally.immobilized / n if n else 0.0
        eng_moved_f = eng_moved / 100.0

        print(f"\n[{scenario}] Showdown n={n} seeds "
              f"(start {args.seed_start})")
        print(f"    engine    : moved={eng_moved:5.1f}%  immobilized={eng_immob:5.1f}%")
        print(f"    showdown  : moved={100*moved_frac:5.1f}%  immobilized={100*immob_frac:5.1f}%  "
              f"(cant Attract={tally.cant_attract}, cant par={tally.cant_par}, "
              f"moved={tally.moved}, other={tally.other})")
        print(f"    activate  : |-activate|..|move: Attract seen in {tally.activate_seen}/{n} measured turns")
        if sample:
            print("    sample p2 lines (seed {}):".format(args.seed_start))
            for l in sample:
                print("        " + l)

        # Assertions.
        if n < args.seeds * 0.9:
            failures.append(f"{scenario}: only {n}/{args.seeds} seeds produced a measured turn")
        if tally.activate_seen != n:
            failures.append(f"{scenario}: activate line missing in {n - tally.activate_seen}/{n} turns")
        if not _binom_ok(moved_frac, eng_moved_f, n):
            failures.append(
                f"{scenario}: Showdown moved {100*moved_frac:.1f}% not within 4sigma of "
                f"engine {eng_moved:.1f}%")
        # Both branches must actually occur (the bug was 0% immobilize).
        if tally.moved == 0:
            failures.append(f"{scenario}: never observed a move branch")
        if tally.immobilized == 0:
            failures.append(f"{scenario}: never observed an immobilize branch (the silent-no-op bug)")
        if scenario == "para":
            # Composition: BOTH cant Attract AND cant par must appear.
            if tally.cant_attract == 0:
                failures.append("para: no 'cant Attract' observed")
            if tally.cant_par == 0:
                failures.append("para: no 'cant par' observed (paralysis composition unverified)")

    print("\n" + "=" * 78)
    if failures:
        print("DIFFERENTIAL FAILED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("DIFFERENTIAL PASSED: Showdown ground truth matches the patched engine "
          "on move/immobilize branch probabilities and protocol shapes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
