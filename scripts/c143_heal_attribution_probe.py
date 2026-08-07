#!/usr/bin/env python
"""C143 — every measurement behind `reports/c143_heal_attribution_diagnosis.md`.

Three independent measurements, all reproducible from a clean checkout:

1. ``showdown`` — three GENERATED gen3 Custom Game boundaries (no randbats seed, no
   holdout) that isolate why the seeder's own Leftovers tick is missing from
   `19200244/115`'s protocol. Gen 3 inherits gen 4's residual ordering, where
   Leftovers is ``onResidualOrder 10 / subOrder 4`` and the Leech Seed volatile is
   ``10 / subOrder 5`` — ONE speed-sorted bucket per Pokemon, not two global phases.
   ``sim/battle.ts`` ends the bucket with ``this.faintMessages(); if (this.ended)
   return;``, so a SLOWER seeder whose capped drain kills the opponent's LAST
   Pokemon never reaches its own Leftovers slot.

     A  slow seeder, victim IS the opponent's last mon  -> seeder Leftovers ABSENT
     B  slow seeder, victim is NOT the last mon         -> seeder Leftovers PRESENT
     C  fast seeder, victim IS the last mon             -> seeder Leftovers PRESENT

   A and B differ in ONE bit: whether a spare Pokemon sits behind the victim.

2. ``engine`` — the crate's branch renderer on the SAME two states as A and B. The
   HP arithmetic is identical to Showdown's in both; only the attribution differs,
   and only in A, where the bare silent drain comes back tagged
   ``[from] item: Leftovers``.

3. ``matrix`` — the recorded `19200244/115` row replayed against every roll its own
   damage fan can throw in the residual-lethal band, crossed with candidate
   collapsed representatives, through the UNMODIFIED shipped
   ``evaluate_boundary_strict``. Separates the renderer defect from the G8
   representative defect by measurement rather than by argument.

The row's artifact is not on `main` (it lands with the C141 PR), so ``--row`` takes
the path to the sweep JSON. Nothing here re-runs Showdown on any seed at or above
19,200,000: the row is read from the committed artifact and replayed against the
local engine, and every Showdown call is a generated Custom Game fixture.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

SEED = 19200244
STEP = 115

# --- part 1: generated Showdown boundaries ----------------------------------

_TURNS = (
    [("move leechseed", "move seismictoss")]
    + [("move seismictoss", "move seismictoss")] * 4
    + [("move splash", "move seismictoss")]
)


def _variants():
    from pokezero.showdown_fixture import FixturePokemon

    def seeder(species, ability):
        return FixturePokemon(
            species=species, ability=ability, item="Leftovers",
            moves=("Leech Seed", "Seismic Toss", "Splash"),
        )

    victim = FixturePokemon(
        species="Blissey", ability="Natural Cure", item="Leftovers",
        moves=("Seismic Toss", "Splash"),
    )
    filler = FixturePokemon(
        species="Misdreavus", ability="Levitate", item="None", moves=("Splash",),
    )
    return {
        # name -> (p1 team, p2 team, predicted seeder-Leftovers disposition)
        "A_slow_seeder_victim_is_last": ([seeder("Snorlax", "Immunity")], [victim], "absent"),
        "B_slow_seeder_victim_not_last": (
            [seeder("Snorlax", "Immunity")], [victim, filler], "after_mirror",
        ),
        "C_fast_seeder_victim_is_last": ([seeder("Suicune", "Pressure")], [victim], "before_drain"),
    }


def measure_showdown(seed: int) -> dict:
    from pokezero.showdown_fixture import run_multi_turn_fixture

    out = {}
    for name, (p1, p2, predicted) in _variants().items():
        result = run_multi_turn_fixture(p1_team=p1, p2_team=p2, turns=_TURNS, seed=seed)
        if result.error_lines:
            raise SystemExit(f"{name}: showdown reported errors {result.error_lines}")
        # Drop `|t:|<unix>`: it is wall-clock and would make the artifact
        # differ on every run for no measured reason.
        lines = [l for l in result.steps[-1].protocol_lines if not l.startswith("|t:|")]
        seeder_name = p1[0].species
        drain_at = next(
            (i for i, line in enumerate(lines) if "[from] Leech Seed" in line), None
        )
        lefto_at = next(
            (
                i for i, line in enumerate(lines)
                if line.startswith(f"|-heal|p1a: {seeder_name}|")
                and "[from] item: Leftovers" in line
            ),
            None,
        )
        if lefto_at is None:
            observed = "absent"
        elif drain_at is not None and lefto_at < drain_at:
            observed = "before_drain"
        else:
            observed = "after_mirror"
        out[name] = {
            "protocol": lines,
            "predicted_seeder_leftovers": predicted,
            "observed_seeder_leftovers": observed,
            "agrees": observed == predicted,
            "terminal": result.terminal,
        }
    return out


# --- part 2: the crate renderer on the same two positions -------------------

def measure_engine() -> dict:
    import pokezero_search
    from pokezero.poke_engine_adapter import (
        BattleSpec, MoveSpec, PokemonSpec, SideSpec, build_poke_engine_state,
    )

    def mon(ident, types, hp, maxhp, spe, moves, item=None, ability=None):
        return PokemonSpec(
            id=ident, level=100, types=types, hp=hp, maxhp=maxhp, attack=100, defense=100,
            special_attack=100, special_defense=100, speed=spe, item=item, ability=ability,
            moves=tuple(MoveSpec(id=m, pp=32) for m in moves),
        )

    def side(mons, volatiles=()):
        return SideSpec(
            pokemon=tuple(mons), active_index=0, side_conditions={}, boosts={},
            volatile_statuses=tuple(volatiles), volatile_status_durations={},
        )

    def snorlax():
        return mon("snorlax", ("normal",), 461, 461, 96,
                   ("leechseed", "seismictoss", "splash"), item="leftovers", ability="immunity")

    def blissey():
        return mon("blissey", ("normal",), 6, 651, 146, ("seismictoss", "splash"),
                   item="leftovers", ability="naturalcure")

    def filler():
        return mon("misdreavus", ("ghost",), 261, 261, 156, ("splash",), ability="levitate")

    cases = {
        "A_slow_seeder_victim_is_last": ([snorlax()], [blissey()]),
        "B_slow_seeder_victim_not_last": ([snorlax()], [blissey(), filler()]),
    }
    out = {}
    for name, (p1, p2) in cases.items():
        state = build_poke_engine_state(
            BattleSpec(side_one=side(p1), side_two=side(p2, volatiles=("leechseed",)))
        ).to_string()
        ctx = json.dumps({
            "p1": [m.id.title() for m in p1], "p2": [m.id.title() for m in p2], "turn": 7,
        })
        report = json.loads(
            pokezero_search.branch_events(state, "splash", "seismictoss", ctx, True, False)
        )
        branches = report.get("branches") or []
        if len(branches) != 1:
            raise SystemExit(f"{name}: expected one deterministic arm, got {len(branches)}")
        events = list(branches[0]["events"])
        drain = [e for e in events if e.startswith("|-heal|p1a: Snorlax|407/461")]
        out[name] = {
            "events": events,
            "drain_line": drain[0] if drain else None,
            "drain_is_silent": bool(drain) and drain[0].endswith("|[silent]"),
        }
    return out


# --- part 3: the band-versus-representative matrix --------------------------

_MAXHP, _HP0, _PRE_P1 = 407, 157, 219
_CAP, _LEFT = _MAXHP // 8, _MAXHP // 16
# The engine's own 16-roll fan for Flamethrower into this Wigglytuff, from
# poke_engine.calculate_damage; asserted against the live engine below.
_FAN = [135, 136, 138, 139, 141, 143, 144, 146, 147, 149, 151, 152, 154, 155, 157, 159]
_BAND = [d for d in _FAN if d < _HP0]  # every non-KO roll is residual-lethal here
_SHIPPING_REP = 145  # sum(_BAND) // len(_BAND)


def _observation(row: dict, roll: int) -> dict:
    after = _HP0 - roll
    healed = after + _LEFT
    drain = min(_CAP, healed)
    assert healed <= _CAP, roll
    new = copy.deepcopy(row)
    new["protocol"] = [
        "|", "|t:|1786096679",
        "|move|p2a: Wigglytuff|Fire Blast|p1a: Moltres",
        "|-resisted|p1a: Moltres",
        f"|-damage|p1a: Moltres|{_PRE_P1}/268 par",
        "|move|p1a: Moltres|Flamethrower|p2a: Wigglytuff",
        f"|-damage|p2a: Wigglytuff|{after}/407 brn",
        "|",
        f"|-heal|p2a: Wigglytuff|{healed}/407 brn|[from] item: Leftovers",
        "|-damage|p2a: Wigglytuff|0 fnt|[from] Leech Seed|[of] p1a: Moltres",
        f"|-heal|p1a: Moltres|{_PRE_P1 + drain}/268 par|[silent]",
        "|faint|p2a: Wigglytuff", "|", "|win|PokeZero p1",
    ]
    new["observed"] = dict(row["observed"])
    new["observed"]["p1_hp"] = _PRE_P1 + drain
    return new


def _repricer(real, rep: int, silent: bool, stats: dict):
    """Re-price the single non-KO Flamethrower arm to `rep`; arm count untouched.

    Keys on the `p1a:`/`p2a:` prefixes rather than on species names: the replay
    path renders side one's active as `unknown5`, and an earlier version of this
    probe keyed on `Moltres` and silently rewrote nothing. `stats` exists so the
    control cannot be vacuous.
    """
    after, healed = _HP0 - rep, _HP0 - rep + _LEFT
    drain = min(_CAP, healed)
    assert healed <= _CAP, rep

    def patched(*args, **kwargs):
        report = json.loads(real(*args, **kwargs))
        for branch in report.get("branches") or []:
            events = list(branch["events"])
            if not any(
                e.startswith("|-damage|p2a: ") and e.endswith(f"|{_HP0 - _SHIPPING_REP}/407")
                for e in events
            ):
                continue
            stats["arms"] += 1
            out, p1_hp = [], None
            for event in events:
                if event.startswith("|-damage|p2a: ") and event.endswith(
                    f"|{_HP0 - _SHIPPING_REP}/407"
                ):
                    event = event.replace(f"|{_HP0 - _SHIPPING_REP}/407", f"|{after}/407")
                elif event.startswith("|-heal|p2a: ") and (
                    f"|{_HP0 - _SHIPPING_REP + _LEFT}/407|[from] item: Leftovers" in event
                ):
                    event = event.replace(
                        f"|{_HP0 - _SHIPPING_REP + _LEFT}/407|", f"|{healed}/407|"
                    )
                elif event.startswith("|-damage|p1a: "):
                    p1_hp = int(event.split("|")[3].split("/")[0])
                elif event.startswith("|-heal|p1a: ") and p1_hp is not None:
                    who = event.split("|")[2]
                    tail = "|[silent]" if silent else "|[from] item: Leftovers"
                    # Clamp at maxhp, as the engine does. Without this the rewrite
                    # synthesises `270/268` at rep 135 — physically impossible, and
                    # although it landed only in positive-control cells whose verdict
                    # is set by the diagonal, a probe must not emit states the engine
                    # cannot produce.
                    event = f"|-heal|{who}|{min(268, p1_hp + drain)}/268{tail}"
                    stats["p1_heal_rewrites"] += 1
                    stats.setdefault("lines", []).append(event)
                out.append(event)
            branch["events"] = out
        return json.dumps(report)

    return patched


def measure_matrix(row: dict, reps: list[int]) -> dict:
    import poke_engine
    import pokezero_search
    from cert_sweep_reread import reread_row

    state = poke_engine.State.from_string(row["engine_states"][0])
    maxima = poke_engine.calculate_damage(state, "flamethrower", "fireblast", False)
    fan = sorted({maxima[0][0] * r // 100 for r in range(85, 101)})
    if fan != _FAN:
        raise SystemExit(f"fan drifted: engine says {fan}, probe hard-codes {_FAN}")
    if sum(_BAND) // len(_BAND) != _SHIPPING_REP:
        raise SystemExit("band mean is no longer the shipping representative")

    # A THIRD roll collapse sits on the other side of the same boundary and accounts
    # for the 34.92 % of mass no arm above reaches. Fire Blast into Moltres kills
    # nothing, so no threshold applies and the whole 16-roll fan collapses to its
    # integer mean; the observed roll is the fan's top value.
    fb_max = maxima[1][0]
    fb_rolls = [fb_max * r // 100 for r in range(85, 101)]
    fire_blast = {
        "max": fb_max,
        "fan": sorted(set(fb_rolls)),
        "representative": sum(fb_rolls) // len(fb_rolls),
        "observed_roll": 268 - _PRE_P1,
        "observed_roll_in_fan": (268 - _PRE_P1) in fb_rolls,
        "accuracy_miss_mass_pct": 15.0,
    }

    mirrors = {d: min(_CAP, _HP0 - d + _LEFT) for d in _BAND}
    real = pokezero_search.branch_events
    result = {
        "fan": fan,
        "observed_roll": _HP0 - 11,
        "observed_roll_in_fan": (_HP0 - 11) in fan,
        "residual_lethal_band": _BAND,
        "mirror_heal_by_roll": mirrors,
        "mirror_injective": len(set(mirrors.values())) == len(mirrors),
        "shipping_representative": _SHIPPING_REP,
        "shipping_representative_in_fan": _SHIPPING_REP in fan,
        "shipping_mirror": min(_CAP, _HP0 - _SHIPPING_REP + _LEFT),
        "fire_blast_third_collapse": fire_blast,
        "columns": {},
        "control": {},
    }
    result["shipping_mirror_achievable"] = (
        result["shipping_mirror"] in set(mirrors.values())
    )
    try:
        # Non-vacuous control: the repricer at the shipping representative with the
        # shipping label must reproduce the recorded misses AND must have fired.
        stats = {"arms": 0, "p1_heal_rewrites": 0}
        pokezero_search.branch_events = _repricer(real, _SHIPPING_REP, False, stats)
        _, misses, _ = reread_row(_observation(row, _HP0 - 11))
        result["control"] = {
            "misses_identical_to_recorded": list(misses) == list(row["branch_misses"]),
            "arms_touched": stats["arms"],
            "p1_heal_lines_rewritten": stats["p1_heal_rewrites"],
        }
        saturating = []
        for rep in reps:
            cell = {}
            for silent in (False, True):
                stats = {"arms": 0, "p1_heal_rewrites": 0}
                pokezero_search.branch_events = _repricer(real, rep, silent, stats)
                matched = [
                    roll for roll in _BAND
                    if reread_row(_observation(row, roll))[0] == "matched"
                ]
                cell["renderer_fixed" if silent else "renderer_shipping"] = matched
                # Census, not inference: a representative saturates iff the repricer
                # actually wrote a `268/268` line for it.
                if silent and any("|268/268|" in line for line in stats.get("lines", [])):
                    saturating.append(rep)
            cell["matched_count_renderer_fixed"] = len(cell["renderer_fixed"])
            result["columns"][str(rep)] = cell
        result["saturating_representatives"] = sorted(saturating)
        result["representatives_tested"] = sorted(reps)
        # The saturation claim is a census only if every band member was a row.
        result["band_fully_covered"] = set(_BAND) <= set(reps)
        # The principled "snap the off-fan representative to the nearest fan member"
        # rule is a TIE here: 144 and 146 are both distance 1 from 145.
        result["nearest_fan_members_to_the_shipping_representative"] = sorted(
            d for d in _BAND if abs(d - _SHIPPING_REP) == min(
                abs(x - _SHIPPING_REP) for x in _BAND
            )
        )
    finally:
        pokezero_search.branch_events = real
    return result


# --- part 4: the enumeration oracle, with and without a modelled G33b gate --

_OBSERVED_ROLL = 146
_TRUE_LEFTOVERS_TICK = 268 // 16  # 16 — a genuine Moltres tick, which the gate must not silence


def _is_truncated(events: list[str]) -> bool:
    """The arm's residual phase was cut short by the opposing active's faint.

    Read off the render, not predicted: an arm that ends the battle carries a
    ``|faint|p2a:`` and NO ``|turn|`` line, because `finish_ply` only emits the turn
    marker when the battle continues.
    """
    return (
        any(e.startswith("|faint|p2a: ") for e in events)
        and not any(e.startswith("|turn|") for e in events)
    )


def _g33b_gate(real, pre_p1_hp: int, log: dict):
    """Model the G33b fix at the renderer's output: in an arm truncated by the
    opposing active's faint, side one's Leftovers-tagged heal is the bare drain.

    The strict path compares rendered components only, so rewriting the render
    models the crate change faithfully — the method c140 §6a used.
    """
    def patched(*args, **kwargs):
        report = json.loads(real(*args, **kwargs))
        for branch in report.get("branches") or []:
            events = list(branch["events"])
            if not _is_truncated(events):
                continue
            hp, out = pre_p1_hp, []
            for event in events:
                if event.startswith(("|-damage|p1a: ", "|-heal|p1a: ")):
                    new_hp = int(event.split("|")[3].split("/")[0])
                    if event.startswith("|-heal|p1a: ") and "[from] item: Leftovers" in event:
                        log["deltas"].append(new_hp - hp)
                        event = "|".join(event.split("|")[:4]) + "|[silent]"
                    hp = new_hp
                out.append(event)
            branch["events"] = out
        return json.dumps(report)

    return patched


def measure_enumerated(row: dict) -> dict:
    """Part 4. Must run in a process started with POKEZERO_ENUMERATE_ROLLS=1."""
    import os

    import pokezero_search
    from cert_sweep_reread import reread_row

    if os.environ.get("POKEZERO_ENUMERATE_ROLLS") != "1":
        raise SystemExit(
            "part 4 requires POKEZERO_ENUMERATE_ROLLS=1 in the environment at process start"
        )
    real = pokezero_search.branch_events
    pre_p1_hp = int(row["pre_features"]["p1_hp"])
    out: dict = {"flag": "POKEZERO_ENUMERATE_ROLLS=1"}

    verdict, misses, total = reread_row(row)
    out["shipped_renderer"] = {
        "verdict": verdict,
        "branches": total,
        "misses": len(misses),
        # The oracle reaches the observed magnitude and fails on the LABEL alone.
        "label_only_miss": [
            m for m in misses
            if f"observed_only=[('heal', {_OBSERVED_ROLL - 110})]" in m
            and f"engine_only=[('itemleftovers', {_OBSERVED_ROLL - 110})]" in m
        ],
    }
    # Mass of the arm that reproduces the recorded protocol's HP trace exactly.
    ctx = json.dumps({
        "p1": row["party_display"]["p1"], "p2": row["party_display"]["p2"], "turn": row["turn"],
    })
    report = json.loads(
        real(row["engine_states"][0], row["choices"]["p1"], row["choices"]["p2"], ctx, True, True)
    )
    def _mass(predicate):
        arms = [b for b in report.get("branches") or [] if predicate(b["events"])]
        return {"count": len(arms), "mass_pct": round(sum(float(b["percentage"]) for b in arms), 4)}

    def _flamethrower_matches(events):
        return (
            f"|-damage|p2a: Wigglytuff|{_HP0 - _OBSERVED_ROLL}/407" in events
            and f"|-heal|p2a: Wigglytuff|{_HP0 - _OBSERVED_ROLL + _LEFT}/407|"
            "[from] item: Leftovers" in events
            and _is_truncated(events)
        )

    # Two different questions, kept apart because a single "the oracle emits the
    # observed row" figure conflates them. The first counts arms agreeing with the
    # observed FLAMETHROWER roll (summed over the paralysis and crit splits on the
    # other side of the field); the second additionally requires the observed FIRE
    # BLAST roll, and is the arm the label-only miss is reported at.
    out["arms_reproducing_the_observed_flamethrower_roll"] = _mass(_flamethrower_matches)
    out["arms_reproducing_the_full_observed_trace"] = _mass(
        lambda e: _flamethrower_matches(e) and f"|-damage|p1a: Moltres|{_PRE_P1}/268" in e
    )

    log = {"deltas": []}
    try:
        pokezero_search.branch_events = _g33b_gate(real, pre_p1_hp, log)
        verdict, misses, total = reread_row(row)
    finally:
        pokezero_search.branch_events = real
    deltas = log["deltas"]
    out["modelled_g33b_gate"] = {
        "verdict": verdict,
        "branches": total,
        "misses": len(misses),
        "first_misses": list(misses[:3]),
        "soundness": {
            "heals_relabelled": len(deltas),
            "delta_min": min(deltas) if deltas else None,
            "delta_max": max(deltas) if deltas else None,
            # A genuine Moltres Leftovers tick is exactly 268//16 = 16. If the gate ever
            # silenced one, this would be non-zero and the gate would be over-broad.
            "deltas_equal_to_a_true_leftovers_tick": sum(
                1 for d in deltas if d == _TRUE_LEFTOVERS_TICK
            ),
            "all_deltas_inside_the_residual_lethal_band": bool(deltas) and all(
                min(_BAND_MIRRORS) <= d <= max(_BAND_MIRRORS) for d in deltas
            ),
        },
    }
    return out


_BAND_MIRRORS = [min(_CAP, _HP0 - d + _LEFT) for d in _BAND]


# --- entry point ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=Path, help="sweep JSON containing the 19200244/115 repro")
    parser.add_argument(
        "--enumerated", action="store_true",
        help="run part 4 only; REQUIRES the process to have been started with "
             "POKEZERO_ENUMERATE_ROLLS=1 (the flag is a Rust OnceLock, so one process "
             "is one engine and the two paths cannot be compared in-process)",
    )
    parser.add_argument("--seed", type=int, default=7717, help="fixture seed for the generated boundaries")
    parser.add_argument("--out", type=Path, help="write the artifact here")
    parser.add_argument(
        "--skip-showdown", action="store_true", help="engine-only run (no node bridge)"
    )
    args = parser.parse_args()

    from engine_build_fingerprint import compute_fingerprint

    artifact = {
        "row": f"{SEED}/{STEP}",
        "engine_fingerprint": compute_fingerprint()["fingerprint"],
        "fixture_seed": args.seed,
    }
    if args.enumerated:
        if not args.row:
            raise SystemExit("--enumerated requires --row")
        row = next(
            r for r in json.loads(args.row.read_text())["repros"] if r.get("seed") == SEED
        )
        artifact["enumerated"] = measure_enumerated(row)
        text = json.dumps(artifact, indent=2, sort_keys=True)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
        return
    if not args.skip_showdown:
        artifact["showdown"] = measure_showdown(args.seed)
    artifact["engine"] = measure_engine()
    if args.row:
        row = next(
            r for r in json.loads(args.row.read_text())["repros"] if r.get("seed") == SEED
        )
        # The WHOLE band, plus the off-fan shipping representative: 15 rows x 14 columns.
        artifact["matrix"] = measure_matrix(row, sorted({_SHIPPING_REP, *_BAND}))

    text = json.dumps(artifact, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
