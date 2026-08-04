#!/usr/bin/env python
"""Dedicated differential probe for gen3 Sleep Talk's call fan-out.

Sleep Talk co-occurred with 11 of the 29 structural residue rows, but Sleep Talk
legitimately branches over the move it calls, so a component-count difference is
sometimes correct. Co-occurrence is not evidence; this probe settles it.

GROUND TRUTH, read from the vendored simulator (gen3 inherits gen4, not gen5):

  data/mods/gen3/moves.ts `sleeptalk.onHit` builds its candidate list from the
  user's own move slots, keeping a slot when::

      moveid && !move.flags['nosleeptalk'] && !move.flags['charge']

  then samples it UNIFORMLY. Two gen3-specific details matter:
    * `charge` (two-turn) moves are EXCLUDED — Solar Beam, Fly, Dig, Sky Attack,
      Razor Wind, Skull Bash;
    * if the sampled slot has 0 PP, gen3 emits `|cant|<mon>|nopp|<move>` and the
      turn does NOTHING. It does not resample.

  poke-engine (`State::get_sleep_talk_choices`, src/state.rs:1014) keeps every
  slot except Sleep Talk itself and NONE — no `nosleeptalk` flag test, no
  `charge` test, no PP test.

What this probe reports:

  1. REACHABILITY — how many gen3 randbats sets pair Sleep Talk with a move gen3
     would exclude. A divergence nobody can reach is a footnote; one on the
     randbats distribution is a bug. The `charge` / `nosleeptalk` sets are READ
     from the vendored `data/moves.ts` flags (17 and 40 moves respectively), not
     hardcoded; the report states which source was used. NOTE the scope limit:
     this is SET COMPOSITION only and cannot see the 0-PP arm below, which is a
     STATE condition. Do not quote `reachable: false` as "the divergence is
     unreachable" -- it means "the FLAG arm is unreachable on this variant set".
  2. ENGINE FAN-OUT — the engine's branch count and weights for a curated set,
     against the count gen3's rule prescribes.
  3. SHOWDOWN DIFFERENTIAL — a scripted sleeping Sleep Talk user over N seeds:
     which moves does the real sim actually call?

Usage::

    PYTHONPATH=src python scripts/gen3_sleeptalk_probe.py \\
        --showdown-root <showdown> [--seeds 80]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import poke_engine as pe  # noqa: E402
import pokezero_search  # noqa: E402

from pokezero.local_showdown import LocalShowdownConfig  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, run_multi_turn_fixture  # noqa: E402

# The exclusion sets are READ FROM THE VENDORED SIMULATOR, not hardcoded.
#
# They used to be hardcoded while this file's docstring claimed the rule came from
# `data/mods/gen3/moves.ts` flags. Independent review caught the drift: the
# hardcoded `nosleeptalk` set held 5 moves where the data has 40, missing six
# that a gen3 randbats set could plausibly carry -- `dive`, `bide`, `focuspunch`,
# `uproar`, `mimic`, `sketch`. The recorded "unreachable" conclusion survives
# recomputation with the corrected sets (still 0 affected variants), but the
# false-negative mechanism was live, and a probe whose stated ground truth is a
# file it does not read cannot be trusted the next time the format changes.
#
# Fallbacks are the old hardcoded sets, used only if the data cannot be read; the
# probe reports which source it used so a fallback run is never mistaken for a
# data-backed one.
_FALLBACK_CHARGE = {
    "solarbeam", "fly", "dig", "skyattack", "razorwind", "skullbash", "bounce",
}
_FALLBACK_NOSLEEPTALK = {"sleeptalk", "mirrormove", "assist", "metronome", "naturepower"}


def _move_flag_sets(showdown_root: str) -> tuple[set[str], set[str], str]:
    """`(charge, nosleeptalk, source)` read from the vendored `data/moves.ts`.

    gen3 inherits the base move table; `data/mods/gen3/moves.ts` overrides
    individual entries but does not remove these two flags from the base ones, so
    the base table is the right source for "which moves carry the flag at all".
    """

    import re
    from pathlib import Path

    path = Path(showdown_root) / "data" / "moves.ts"
    try:
        text = path.read_text()
    except OSError:
        return set(_FALLBACK_CHARGE), set(_FALLBACK_NOSLEEPTALK), f"FALLBACK (could not read {path})"

    charge: set[str] = set()
    nosleeptalk: set[str] = set()
    # Top-level entries start at one tab of indentation: `\n\tmoveid: {`.
    for entry in re.split(r"\n\t(?=[a-z0-9]+: \{)", text):
        head = re.match(r"([a-z0-9]+): \{", entry)
        if not head:
            continue
        flags_match = re.search(r"flags: \{([^}]*)\}", entry, re.S)
        if not flags_match:
            continue
        flags = flags_match.group(1)
        if re.search(r"\bcharge\s*:\s*1", flags):
            charge.add(head.group(1))
        if re.search(r"\bnosleeptalk\s*:\s*1", flags):
            nosleeptalk.add(head.group(1))
    if not charge or not nosleeptalk:
        return (set(_FALLBACK_CHARGE), set(_FALLBACK_NOSLEEPTALK),
                f"FALLBACK (parsed 0 flags from {path}; format may have changed)")
    return charge, nosleeptalk, str(path)


def _mon(species, moves, *, ability, item=None, level=80):
    return FixturePokemon(
        species=species, moves=tuple(moves), ability=ability, item=item, level=level,
        evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")},
    )


def _emon(idn, moves, *, maxhp=300, hp=None, status="none"):
    return pe.Pokemon(
        id=idn, level=80, types=("normal", "typeless"), hp=maxhp if hp is None else hp,
        maxhp=maxhp, ability="none", item="none", attack=120, defense=120,
        special_attack=120, special_defense=120, speed=100, status=status,
        moves=[pe.Move(id=m, pp=16) for m in moves],
    )


def reachability(showdown_root: str) -> dict:
    """How many gen3 randbats sets pair Sleep Talk with a gen3-excluded move?

    SCOPE -- read this before quoting `reachable`. This measures SET COMPOSITION
    only: does a rolled variant pair Sleep Talk with a move carrying `charge` or
    `nosleeptalk`. The module docstring names a THIRD divergence, the 0-PP arm
    (gen3 emits `|cant|MON|nopp|MOVE` and does not resample; poke-engine has no PP
    test), and that one is a STATE condition -- reachable from any Sleep Talk
    variant once a slot empties -- so no set-composition scan can see it. A
    `reachable: false` here supports "the charge/nosleeptalk FLAG arm is
    unreachable on this variant set", NOT "the get_sleep_talk_choices divergence
    is unreachable". The returned dict says so in `scope` so a caller cannot quote
    the narrow result as the broad one.
    """

    from pokezero.randbat import Gen3RandbatSource

    charge, nosleeptalk, flag_source = _move_flag_sets(showdown_root)
    excluded = charge | nosleeptalk

    # Real shape (read, not guessed): to_payload()["universes"][species]["variants"]
    # is the list of concrete 4-move sets the generator can produce.
    source = Gen3RandbatSource.from_showdown_root(showdown_root)
    universes = source.to_payload().get("universes", {})
    variants = 0
    with_sleeptalk = 0
    conflicting: Counter = Counter()
    for species, entry in universes.items():
        for variant in entry.get("variants", []) or []:
            moves = variant.get("moves") or []
            if not moves:
                continue
            variants += 1
            pool = {str(m).lower().replace(" ", "").replace("-", "") for m in moves}
            if "sleeptalk" not in pool:
                continue
            with_sleeptalk += 1
            for bad in sorted((pool & excluded) - {"sleeptalk"}):
                conflicting[f"{species}:{bad}"] += 1
    return {
        "variants": variants,
        "variants_with_sleeptalk": with_sleeptalk,
        "sleeptalk_plus_gen3_excluded_move": dict(conflicting),
        "affected_variant_count": sum(conflicting.values()),
        "reachable": bool(conflicting),
        "scope": ("SET COMPOSITION ONLY. `reachable` covers the charge/nosleeptalk "
                  "FLAG arm. The 0-PP arm is a STATE condition and is NOT measured "
                  "here; do not read this as the whole get_sleep_talk_choices "
                  "divergence being unreachable."),
        "flag_source": flag_source,
        "charge_move_count": len(charge),
        "nosleeptalk_move_count": len(nosleeptalk),
    }


def engine_fanout(excluded: set[str] | None = None) -> list[dict]:
    """Engine branch count vs the count gen3's exclusion rule prescribes."""

    cases = [
        ("no_excluded_moves", ["sleeptalk", "bodyslam", "curse", "rest"]),
        ("with_charge_move", ["sleeptalk", "bodyslam", "solarbeam", "rest"]),
        ("two_charge_moves", ["sleeptalk", "solarbeam", "fly", "rest"]),
    ]
    rows = []
    dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
    for name, moves in cases:
        s1 = pe.Side(active_index="0",
                     pokemon=[_emon("snorlax", moves, hp=150, status="sleep")] + [dummy] * 5)
        s2 = pe.Side(active_index="0", pokemon=[_emon("starmie", ["splash", "tackle"])] + [dummy] * 5)
        state = pe.State(side_one=s1, side_two=s2, weather="none", terrain="none",
                         trick_room=False)
        # Branch COUNT is not call count: each called move fans out further over
        # its own chance nodes (crit, secondary, accuracy). Attribute branches to
        # the move that was called, via the mapper's rendered `[from] Sleep Talk`
        # line, and sum probability per move — that is what compares to gen3's
        # uniform 1/n.
        ctx = json.dumps({"p1": ["Snorlax"], "p2": ["Starmie"], "turn": 3})
        try:
            rendered = json.loads(
                pokezero_search.branch_events(
                    state.to_string(), "sleeptalk", "splash", ctx, True, True
                )
            )
        except BaseException as error:  # noqa: BLE001
            rows.append({"case": name, "error": f"{type(error).__name__}: {error}"})
            continue
        per_move: Counter = Counter()
        for branch in rendered.get("branches") or []:
            pct = float(branch.get("percentage") or 0.0)
            called = None
            for line in branch.get("events") or []:
                if "|move|p1a" in line and "Sleep Talk" in line and "[from]" in line:
                    called = line.split("|")[3].strip().lower().replace(" ", "")
                    break
            per_move[called or "(no call rendered)"] += pct
        callable_gen3 = [m for m in moves if m not in (excluded or set())]
        rows.append({
            "case": name,
            "moveset": moves,
            "gen3_callable": sorted(callable_gen3),
            "gen3_expected_share_each": round(100.0 / len(callable_gen3), 2) if callable_gen3 else None,
            "engine_called_moves": sorted(k for k in per_move if not k.startswith("(")),
            "engine_share_per_move": {k: round(v, 2) for k, v in sorted(per_move.items())},
        })
    return rows


def showdown_calls(showdown_root: str, seeds: int, seed_start: int) -> dict:
    """Which moves does the real sim actually call, over N seeds?"""

    config = LocalShowdownConfig(showdown_root=showdown_root)
    # Snorlax: Sleep Talk + Solar Beam (charge, gen3-EXCLUDED) + two callable.
    sleeper = _mon("Snorlax", ["Sleep Talk", "Solar Beam", "Body Slam", "Rest"],
                   ability="Immunity")
    poker = _mon("Skarmory", ["Drill Peck", "Spikes", "Roar", "Protect"], ability="Keen Eye")

    # p1 must ATTACK: Rest at full HP FAILS, so a protecting p1 never puts the
    # sleeper to sleep and Sleep Talk never fires (this probe's first revision
    # measured 60 seeds of nothing).
    turns = [("move drillpeck", "move rest")] + [("move drillpeck", "move sleeptalk")] * 3
    called: Counter = Counter()
    usable = 0
    for offset in range(seeds):
        try:
            result = run_multi_turn_fixture(
                p1_team=[poker], p2_team=[sleeper],
                turns=turns, seed=seed_start + offset, config=config,
            )
        except Exception:  # noqa: BLE001
            continue
        if result.error_lines or len(result.steps) < 2:
            continue
        usable += 1
        for step in result.steps[1:]:
            for line in step.protocol_lines:
                if "|move|p2a" in line and "[from]" in line and "Sleep Talk" in line:
                    called[line.split("|")[3].strip().lower().replace(" ", "")] += 1
                if "|cant|p2a" in line and "nopp" in line:
                    called["(cant: nopp)"] += 1
    return {"seeds_usable": usable, "calls": dict(called)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--showdown-root", required=True)
    ap.add_argument("--seeds", type=int, default=80)
    ap.add_argument("--seed-start", type=int, default=1400000)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    report = {
        "gen3_rule": (
            "data/mods/gen3/moves.ts sleeptalk.onHit: candidates = own move slots with "
            "!flags['nosleeptalk'] && !flags['charge'], sampled UNIFORMLY; a sampled slot "
            "at 0 PP emits |cant|..|nopp| and does nothing (no resample)."
        ),
        "engine_rule": (
            "State::get_sleep_talk_choices (src/state.rs:1014): every slot except "
            "SLEEPTALK and NONE. No nosleeptalk test, no charge test, no PP test."
        ),
        "reachability": reachability(args.showdown_root),
        "engine_fanout": engine_fanout(
            set().union(*_move_flag_sets(args.showdown_root)[:2])
        ),
        "showdown_calls": showdown_calls(args.showdown_root, args.seeds, args.seed_start),
    }

    print("=" * 90)
    print("gen3 rule :", report["gen3_rule"])
    print("engine    :", report["engine_rule"])
    print()
    print("-- reachability in gen3 randbats --")
    print(" ", json.dumps(report["reachability"])[:400])
    print()
    print("-- engine fan-out vs gen3 --")
    for row in report["engine_fanout"]:
        if "error" in row:
            print(f"  {row['case']}: ERROR {row['error']}")
            continue
        extra = sorted(set(row["engine_called_moves"]) - set(row["gen3_callable"]))
        missing = sorted(set(row["gen3_callable"]) - set(row["engine_called_moves"]))
        flag = "MISMATCH" if (extra or missing) else "ok"
        print(f"  {row['case']:<20} moveset={row['moveset']}")
        print(f"  {'':<20} gen3 callable={row['gen3_callable']} "
              f"({row['gen3_expected_share_each']}% each)")
        print(f"  {'':<20} engine called ={row['engine_called_moves']}")
        print(f"  {'':<20} engine shares ={row['engine_share_per_move']}")
        if extra:
            print(f"  {'':<20} ENGINE CALLS MOVES GEN3 EXCLUDES: {extra}")
        if missing:
            print(f"  {'':<20} ENGINE NEVER CALLS: {missing}")
        print(f"  {'':<20} [{flag}]")
    print()
    print("-- Showdown actual calls (Sleep Talk + Solar Beam + Body Slam + Rest) --")
    print(" ", json.dumps(report["showdown_calls"]))

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
