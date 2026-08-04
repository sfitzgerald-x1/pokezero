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
    * `charge` (two-turn) moves are EXCLUDED. 17 moves in the gen3 Dex table carry
      the flag, of which 8 are gen3-LEGAL: Dig, Dive, Fly, Bounce, Skull Bash,
      Razor Wind, Sky Attack, Solar Beam. The other 9 are `isNonstandard: Future`
      entries the gen3 table retains; a gen3 set cannot contain them, and keeping
      them in the membership test is the safer direction. (An earlier version of
      this line listed six and omitted Dive and Bounce.);
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
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import poke_engine as pe  # noqa: E402
import pokezero_search  # noqa: E402

from pokezero.local_showdown import LocalShowdownConfig  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, run_multi_turn_fixture  # noqa: E402

# The exclusion sets come from the SIMULATOR'S OWN RESOLVER, for gen3.
#
# Two wrong answers preceded this one, both found by independent review:
#
#  1. Originally hardcoded, while the docstring claimed the flags came from
#     `data/mods/gen3/moves.ts`. The hardcoded `nosleeptalk` set held 5 where
#     gen3 has 35, and it wrongly INCLUDED `naturepower`.
#  2. Then parsed out of base `data/moves.ts`, which gives 40 -- the gen9 answer,
#     not gen3's. A mod entry's `flags: {...}` REPLACES the parent's wholesale,
#     and gen3 inherits through gen4-gen8: `data/mods/gen5/moves.ts` drops
#     `nosleeptalk` from `fly`, and `data/mods/gen4/moves.ts` drops it from
#     `mimic`, `sketch`, `naturepower` and `struggle`. So base-table parsing
#     reports Mimic and Sketch as gen3-excluded when they are not.
#
# gen3's `sleeptalk.onHit` reads `this.dex.moves.get(moveid).flags` -- GEN3-resolved
# flags. The only thing that gets that right without reimplementing the mod chain
# is the simulator's own `Dex`, which `randbat.py` already requires to be built
# (`_AUDIT_ENGINE_RELATIVE_PATHS` lists `dist/sim`). Ask it.
#
# gen3 truth as of 2026-08-03, read from `Dex.forFormat("gen3randombattle")`:
#   charge       17 in the gen3 Dex table (8 of them gen3-legal)
#   nosleeptalk  35 in the gen3 Dex table (14 of them gen3-legal)
# The rest are `isNonstandard: Future` entries the gen3 table retains. A gen3 set
# cannot contain them, so the unfiltered set is the safer membership test -- but
# "gen3 has 35" would be the same species of overstatement this file is fixing.
# The snapshots below are that output, used only when node/dist is unavailable.
# When the Dex IS available its answer wins and any difference from the snapshot
# is reported as drift rather than silently absorbed.
_GEN3_CHARGE_SNAPSHOT = {
    "bounce", "dig", "dive", "electroshot", "fly", "freezeshock", "geomancy",
    "iceburn", "meteorbeam", "phantomforce", "razorwind", "shadowforce",
    "skullbash", "skyattack", "skydrop", "solarbeam", "solarblade",
}
_GEN3_NOSLEEPTALK_SNAPSHOT = {
    "assist", "beakblast", "belch", "bide", "blazingtorque", "bounce", "celebrate",
    "chatter", "combattorque", "copycat", "dig", "dive", "dynamaxcannon",
    "focuspunch", "freezeshock", "geomancy", "holdhands", "iceburn",
    "magicaltorque", "mefirst", "metronome", "mirrormove", "noxioustorque",
    "phantomforce", "razorwind", "shadowforce", "shelltrap", "skullbash",
    "skyattack", "skydrop", "sleeptalk", "solarbeam", "solarblade", "uproar",
    "wickedtorque",
}

_DEX_QUERY = """
const {Dex} = require(process.argv[1]);
const FORMAT = "gen3randombattle";
// ASSERT what the resolver actually gave us. `Dex.forFormat` on an UNKNOWN format
// does not raise: `this.formats.get(name)` returns a non-existent Format whose
// `.mod` is the string "gen9", so `dexes[mod || BASE_MOD]` hands back the gen9
// table -- 17 charge / 40 nosleeptalk, exactly the wrong answer this file exists
// to stop reporting. A format rename upstream, a typo, or a stale
// dist/config/formats.js would silently produce it. So verify the mapping exists
// AND that the dex we got is gen3, and crash otherwise.
if (!Dex.formats.get(FORMAT).exists) {
  throw new Error(`format ${FORMAT} does not exist in this build; ` +
                  `Dex.forFormat would silently fall back to the base (gen9) mod`);
}
const d = Dex.forFormat(FORMAT);
if (d.gen !== 3 || d.currentMod !== "gen3") {
  throw new Error(`resolver gave gen${d.gen}/${d.currentMod}, expected gen3/gen3`);
}
const charge = [], nost = [];
for (const m of d.moves.all()) {
  if (m.flags && m.flags.charge) charge.push(m.id);
  if (m.flags && m.flags.nosleeptalk) nost.push(m.id);
}
process.stdout.write(JSON.stringify({gen: d.gen, currentMod: d.currentMod, charge, nosleeptalk: nost}));
"""


def _move_flag_sets(showdown_root: str) -> tuple[set[str], set[str], str]:
    """`(charge, nosleeptalk, source)` for GEN3, from the simulator's `Dex`.

    Falls back to the dated snapshots above only when node or `dist/sim/dex.js`
    is unavailable, and says so in `source` so a fallback run is never mistaken
    for a resolver-backed one.
    """

    dex_js = Path(showdown_root) / "dist" / "sim" / "dex.js"
    if not dex_js.is_file():
        return (set(_GEN3_CHARGE_SNAPSHOT), set(_GEN3_NOSLEEPTALK_SNAPSHOT),
                f"SNAPSHOT (no {dex_js}; run `npx tsc` in the showdown checkout)")
    try:
        # check=False on purpose: with check=True the CalledProcessError message
        # embeds the whole query and says only "non-zero exit status 1", losing
        # node's actual error -- the one useful part.
        proc = subprocess.run(
            ["node", "-e", _DEX_QUERY, str(dex_js)],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if proc.returncode != 0:
            # Pick the DIAGNOSIS line, not the last line. For an uncaught throw
            # under `node -e`, stderr ends with the version banner ("Node.js
            # v22.x"), so `[-1]` reported that and nothing else -- collapsing the
            # two guards' distinct messages ("format ... does not exist" vs
            # "resolver gave genN/mod") into a version string and leaving a
            # fallback run undiagnosable.
            lines = (proc.stderr or "").strip().splitlines()
            why = (
                next((ln.strip() for ln in lines if "Error: " in ln), None)
                or " | ".join(ln.strip() for ln in lines[-3:])
                or f"exit {proc.returncode}"
            )
            return (set(_GEN3_CHARGE_SNAPSHOT), set(_GEN3_NOSLEEPTALK_SNAPSHOT),
                    f"SNAPSHOT (Dex query failed: {why[:200]})")
        payload = json.loads(proc.stdout)
        # DELIBERATELY REDUNDANT with the JS assertions, and unreachable while they
        # stand: this only fires if a future edit removes or weakens them. Review
        # confirmed by mutation that it does catch that case, so it is not dead
        # code -- but do not read its presence as coverage of the JS guards.
        if payload.get("gen") != 3 or payload.get("currentMod") != "gen3":
            return (set(_GEN3_CHARGE_SNAPSHOT), set(_GEN3_NOSLEEPTALK_SNAPSHOT),
                    f"SNAPSHOT (resolver reported gen{payload.get('gen')}/"
                    f"{payload.get('currentMod')}, not gen3/gen3)")
        charge = {str(m) for m in payload["charge"]}
        nosleeptalk = {str(m) for m in payload["nosleeptalk"]}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        return (set(_GEN3_CHARGE_SNAPSHOT), set(_GEN3_NOSLEEPTALK_SNAPSHOT),
                f"SNAPSHOT (Dex query failed: {type(exc).__name__}: {exc})")

    # A resolver that answers with an implausibly small set is the dangerous
    # case: it would be published with a resolver-backed provenance stamp while
    # most exclusions silently vanished -- the exact false negative this whole
    # file exists to prevent. Refuse to call that data-backed.
    # SUBSET, not cardinality. A same-size SUBSTITUTION -- e.g. `dive` replaced by
    # a bogus id -- keeps the count at 35 while silently losing a real gen3
    # exclusion, and a length check cannot see it. Growth is deliberately NOT
    # gated: a legitimate upstream addition should be reported as drift with the
    # resolver winning.
    lost_charge = _GEN3_CHARGE_SNAPSHOT - charge
    lost_nost = _GEN3_NOSLEEPTALK_SNAPSHOT - nosleeptalk
    if lost_charge or lost_nost:
        missing = []
        if lost_charge:
            missing.append(f"charge missing {sorted(lost_charge)}")
        if lost_nost:
            missing.append(f"nosleeptalk missing {sorted(lost_nost)}")
        return (charge | _GEN3_CHARGE_SNAPSHOT, nosleeptalk | _GEN3_NOSLEEPTALK_SNAPSHOT,
                f"UNION of Dex and SNAPSHOT -- the resolver LOST moves the "
                f"2026-08-03 snapshot has ({'; '.join(missing)}); treat as SUSPECT "
                f"and re-derive. Counts below are UNION sizes, so neither the "
                f"resolver's answer nor the snapshot's")
    source = str(dex_js)
    drift = ((charge ^ _GEN3_CHARGE_SNAPSHOT) | (nosleeptalk ^ _GEN3_NOSLEEPTALK_SNAPSHOT))
    if drift:
        source += f" (DRIFT vs 2026-08-03 snapshot: {sorted(drift)} -- Dex wins; update the snapshot)"
    return charge, nosleeptalk, source


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


def reachability(
    showdown_root: str,
    charge: set[str],
    nosleeptalk: set[str],
    flag_source: str,
) -> dict:
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

    excluded = charge | nosleeptalk

    # Real shape (read, not guessed): to_payload()["universes"][species]["variants"]
    # is the list of concrete 4-move sets the generator can produce.
    source = Gen3RandbatSource.from_showdown_root(showdown_root)
    universes = source.to_payload().get("universes", {})
    variants = 0
    with_sleeptalk = 0
    affected_variants = 0
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
            bad_here = sorted((pool & excluded) - {"sleeptalk"})
            if bad_here:
                affected_variants += 1
            for bad in bad_here:
                conflicting[f"{species}:{bad}"] += 1
    return {
        "variants": variants,
        "variants_with_sleeptalk": with_sleeptalk,
        "sleeptalk_plus_gen3_excluded_move": dict(conflicting),
        # DISTINCT variants. This used to be sum(conflicting.values()), which
        # counts (species, move) PAIRINGS -- a variant pairing Sleep Talk with two
        # excluded moves was counted twice and the field overstated variants.
        "affected_variant_count": affected_variants,
        "affected_pairing_count": sum(conflicting.values()),
        "reachable": bool(conflicting),
        "scope": (
            "SET COMPOSITION ONLY. `reachable` covers the charge/nosleeptalk FLAG "
            "arm and nothing else. TWO further divergences are STATE conditions "
            "that no composition scan can see, and both are reachable on this "
            "variant set: (1) 0 PP -- gen3 emits |cant|MON|nopp|MOVE and does NOT "
            "resample, poke-engine has no PP test, reachable from any of the 70 "
            "Sleep Talk variants once a slot empties; (2) the choicelock/Encore "
            "gate -- gen3's sleeptalk inherits gen4's onTryHit, "
            "`!volatiles.choicelock && !volatiles.encore`, so Sleep Talk FAILS "
            "outright while Encored or Choice-locked, and 95 of 1682 variants "
            "carry Encore. Do NOT read `reachable: false` as the whole "
            "get_sleep_talk_choices divergence being unreachable."
        ),
        "flag_source": flag_source,
        "charge_move_count": len(charge),
        "nosleeptalk_move_count": len(nosleeptalk),
    }


def engine_fanout(excluded: set[str]) -> list[dict]:
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
        callable_gen3 = [m for m in moves if m not in excluded]
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

    # ONE Dex query, threaded into everything below. This genuinely is one query
    # now: `reachability()` used to resolve again, so a `dist/` rebuild or a
    # timeout between the two calls could produce one artifact whose top level
    # said `dex.js` while its `reachability` block said `SNAPSHOT (failed...)`.
    # This file's history is comments asserting things the code did not do.
    charge_moves, nosleeptalk_moves, flag_source = _move_flag_sets(args.showdown_root)

    report = {
        "gen3_rule": (
            "data/mods/gen3/moves.ts sleeptalk.onHit: candidates = own move slots with "
            "!flags['nosleeptalk'] && !flags['charge'], sampled UNIFORMLY; a sampled slot "
            "at 0 PP emits |cant|..|nopp| and does nothing (no resample)."
        ),
        "engine_rule": (
            "State::get_sleep_talk_choices (src/state.rs:1014): every slot except "
            "SLEEPTALK and NONE. No nosleeptalk test, no charge test, no PP test, "
            "and no choicelock/Encore gate -- FOUR divergences, of which this probe "
            "measures only the first two (see reachability.scope)."
        ),
        "flag_source": flag_source,
        "reachability": reachability(
            args.showdown_root, charge_moves, nosleeptalk_moves, flag_source
        ),
        "engine_fanout": engine_fanout(charge_moves | nosleeptalk_moves),
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
