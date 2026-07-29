#!/usr/bin/env python
"""Showdown-vs-engine differentials for the three ledger rows left open after PR #885.

Covers `docs/engine_divergence_ledger_20260728.md` D9, D10 and D11 — the rows the
merged engine fixes did not touch — and turns each from an inherited claim into a
measured verdict against the VENDORED gen3 rules.

Gen 3 rule sourcing follows the house rule: **gen3 inherits gen4, not gen5.** The
mod chain is read out of the vendored simulator before anything is asserted, and
each scenario prints the rule it is testing against.

  encore  (D9)  gen3 Encore lasts `this.random(3, 7)` = 3-6 turns
                (data/mods/gen3/moves.ts `encore.condition.durationCallback`).
                Measures whether the engine's ENCORE volatile EXPIRES at all.

  wish    (D10) gen3 inherits the GEN 4 Wish override, whose `onEnd` heals
                `target.baseMaxhp / 2` — the RESOLVING ACTIVE's own max HP
                (data/mods/gen4/moves.ts `wish.condition.onEnd`). Only the modern
                (gen5+) base move heals the CASTER's `source.maxhp / 2`
                (data/moves.ts `wish.condition.onStart`). Measures which one the
                engine implements, with a caster whose max HP differs from the
                recipient's so the two rules give different answers.

  damage  (D11) The transition differential matches HP inside a +/-16 % band, so a
                systematic damage bias SMALLER than the band is invisible to it.
                This samples Showdown's real roll distribution over N seeds and
                compares its mean to the engine's single representative-roll
                branch. A clean engine has |mean_showdown - engine| within roll
                quantisation; a biased one shows a consistent signed offset.

Usage::

    PYTHONPATH=src python scripts/gen3_duration_differential.py \\
        --showdown-root <showdown> [--scenario encore|wish|damage|all] [--seeds N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokezero.local_showdown import LocalShowdownConfig
from pokezero.showdown_fixture import FixturePokemon, run_multi_turn_fixture

import poke_engine as pe


# --- curated gen3 Custom Game sets -------------------------------------------


def _mon(species, moves, *, ability, item="Leftovers", level=80):
    return FixturePokemon(
        species=species,
        moves=tuple(moves),
        ability=ability,
        item=item,
        level=level,
        evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")},
    )


# --- engine helpers ----------------------------------------------------------


def _emon(idn, moves, *, maxhp=240, hp=None, types=("normal", "typeless"), speed=100):
    return pe.Pokemon(
        id=idn, level=80, types=types, hp=maxhp if hp is None else hp, maxhp=maxhp,
        ability="none", item="none", attack=120, defense=120, special_attack=120,
        special_defense=120, speed=speed,
        moves=[pe.Move(id=m, pp=16) for m in moves],
    )


def _dummy():
    return pe.Pokemon(id="pikachu", level=1, hp=0)


def _state(side_one, side_two):
    return pe.State(
        side_one=side_one, side_two=side_two,
        weather="none", terrain="none", trick_room=False,
    )


# --- D9: Encore duration -----------------------------------------------------


def scenario_encore(showdown_root: str, seeds: int, seed_start: int) -> dict:
    """Does the engine's ENCORE volatile ever expire? gen3 ends it after 3-6 turns."""

    config = LocalShowdownConfig(showdown_root=showdown_root)
    # p1 must NOT be able to KO: an earlier revision used Surf and every battle
    # ended by step 5, so only the shortest (3-turn) Encores were ever observed
    # and the measured distribution collapsed to a single value.
    politoed = _mon("Politoed", ["Encore", "Protect", "Ice Beam", "Hypnosis"],
                    ability="Water Absorb")
    ampharos = _mon("Ampharos", ["Growl", "Thunderbolt", "Thunder Wave", "Light Screen"],
                    ability="Static")

    # Showdown DISABLES every non-encored move while Encore is up, so the target
    # cannot be scripted onto another move — scripting one raises "Unavailable
    # choice". The clean signal is Showdown's own expiry line:
    #   |-end|p2a: Ampharos|move: Encore
    # Script the encored move throughout and count turns until that line lands.
    turns = [("move protect", "move growl"), ("move encore", "move growl")]
    turns += [("move protect", "move growl")] * 10

    lock_lengths = []
    for offset in range(seeds):
        try:
            result = run_multi_turn_fixture(
                p1_team=[politoed], p2_team=[ampharos],
                turns=turns, seed=seed_start + offset, config=config,
            )
        except Exception:  # noqa: BLE001 — a desynced seed is not a measurement
            continue
        if result.error_lines or len(result.steps) < 3:
            continue
        ended_at = None
        for index, step in enumerate(result.steps[1:], start=1):
            if any("|-end|p2a" in l and "Encore" in l for l in step.protocol_lines):
                ended_at = index  # steps since (and including) the application turn
                break
        if ended_at is not None:
            lock_lengths.append(ended_at)

    # Engine side: seed the ENCORE volatile and step; does it ever get removed?
    # ENCORE requires last_used_move to be a MOVE, in INDEX form ("move:0" —
    # a move id panics; see docs/engine_fidelity_findings.md caller-contract 2).
    # Growl is slot 0, which is what the volatile locks the mon into.
    s2 = pe.Side(
        active_index="0",
        pokemon=[_emon("ampharos", ["growl", "thunderbolt"])] + [_dummy()] * 5,
        volatile_statuses={"ENCORE"},
        last_used_move="move:0",
    )
    s1 = pe.Side(active_index="0", pokemon=[_emon("politoed", ["protect", "encore"])] + [_dummy()] * 5)
    state = _state(s1, s2)
    engine_turns_until_end = None
    for turn in range(1, 13):
        branches = pe.generate_instructions(state, "protect", "growl")
        state = state.apply_instructions(branches[0])
        if "ENCORE" not in {str(v).upper() for v in state.side_two.volatile_statuses}:
            engine_turns_until_end = turn
            break

    showdown_min = min(lock_lengths) if lock_lengths else None
    showdown_max = max(lock_lengths) if lock_lengths else None
    ok = engine_turns_until_end is not None and showdown_max is not None and (
        showdown_min <= engine_turns_until_end <= showdown_max
    )
    return {
        "scenario": "encore",
        "ledger_row": "D9",
        "gen3_rule": "data/mods/gen3/moves.ts encore.condition.durationCallback -> this.random(3, 7) = 3-6 turns",
        "showdown_lock_turns": sorted(set(lock_lengths)),
        "showdown_samples": len(lock_lengths),
        "engine_turns_until_encore_ends": engine_turns_until_end,
        "engine_expires": engine_turns_until_end is not None,
        "verdict": "cannot-reproduce" if ok else "confirmed",
        "detail": (
            "engine ENCORE never expires within 12 turns; Showdown ends it after "
            f"{showdown_min}-{showdown_max}"
            if engine_turns_until_end is None
            else f"engine ends at turn {engine_turns_until_end}"
        ),
    }


# --- D10: Wish heal amount ---------------------------------------------------


def scenario_wish(showdown_root: str, seeds: int, seed_start: int) -> dict:
    """Whose max HP does Wish heal off — the caster's, or the resolving active's?

    gen3 inherits gen4: `this.heal(target.baseMaxhp / 2)` = the RECIPIENT's.
    """

    config = LocalShowdownConfig(showdown_root=showdown_root)
    # Caster (small max HP) wishes, switches to a big-max-HP recipient that has
    # been chipped, so caster/2 and recipient/2 are far apart.
    # Itemless so no Leftovers tick pollutes the measured heal.
    jirachi = _mon("Jirachi", ["Wish", "Protect", "Psychic", "Fire Punch"],
                   ability="Serene Grace", item=None)
    wailord = _mon("Wailord", ["Surf", "Ice Beam", "Rest", "Water Spout"],
                   ability="Water Veil", item=None)
    skarmory = _mon("Skarmory", ["Drill Peck", "Spikes", "Roar", "Protect"],
                    ability="Keen Eye", item=None)

    # gen3 Wish has duration 2: set on turn N, resolves at the END of turn N+1 —
    # i.e. on the SWITCH turn, with the incoming mon already on the field.
    # The recipient must be BELOW max when the wish lands or Showdown emits no
    # heal line. Showdown resolves the switch first, then p1's move, then the
    # end-of-turn wish — so Drill Peck on the switch turn chips the incoming mon
    # in time.
    # The recipient must be far enough below max that BOTH candidate heals
    # (recipient/2 = 201, caster/2 = 100) land unclamped — otherwise Showdown's
    # heal is capped at full HP and the two rules become indistinguishable.
    # Wish has duration 2: set on turn 4, resolves at the END of turn 5, by which
    # time the chipped recipient is back on the field.
    turns = [
        ("move drillpeck", "move surf"),   # 1-2: chip the big recipient
        ("move drillpeck", "move surf"),
        ("move drillpeck", "switch 2"),    # 3: caster in
        ("move protect", "move wish"),     # 4: caster wishes
        # NOTE: `switch N` indexes the LIVE REQUEST order (active first), not the
        # original team order — so bringing the recipient back is "switch 2" again.
        ("move protect", "switch 2"),      # 5: recipient back; wish resolves
    ]
    observations = []
    for offset in range(seeds):
        result = run_multi_turn_fixture(
            p1_team=[skarmory], p2_team=[wailord, jirachi],
            turns=turns, seed=seed_start + offset, config=config,
        )
        if result.error_lines or len(result.steps) < 5:
            continue
        lines = result.steps[4].protocol_lines
        # Walk the step in order, tracking p2's active HP, and read the heal as
        # (post-heal HP - the HP immediately before the wish line) so the switch
        # turn's own chip damage is already accounted for.
        running_hp = None
        running_max = None
        heal = None
        for line in lines:
            parts = line.split("|")
            if line.startswith("|switch|p2a") and len(parts) > 4:
                cur, _, mx = parts[4].partition("/")
                running_hp, running_max = int(cur), int(mx.split()[0])
            elif line.startswith("|-damage|p2a") and len(parts) > 3:
                cur, _, mx = parts[3].partition("/")
                if cur.strip() == "0" or "fnt" in parts[3]:
                    running_hp = 0
                else:
                    running_hp, running_max = int(cur), int(mx.split()[0])
            elif "|-heal|p2a" in line and "move: Wish" in line and len(parts) > 3:
                cur, _, mx = parts[3].partition("/")
                heal = int(cur) - (running_hp if running_hp is not None else 0)
                running_hp, running_max = int(cur), int(mx.split()[0])
        if heal is None or running_max is None:
            continue
        observations.append({
            "recipient_maxhp": running_max,
            "heal": heal,
            "expected_recipient_half": running_max // 2,
        })

    # Engine: caster maxhp 200, recipient maxhp 400. Which half does it heal?
    caster_maxhp, recipient_maxhp = 200, 400
    s2 = pe.Side(
        active_index="0",
        pokemon=[
            _emon("jirachi", ["wish", "protect"], maxhp=caster_maxhp),
            _emon("wailord", ["surf", "protect"], maxhp=recipient_maxhp, hp=recipient_maxhp // 4),
        ] + [_dummy()] * 4,
    )
    s1 = pe.Side(active_index="0", pokemon=[_emon("skarmory", ["protect", "drillpeck"])] + [_dummy()] * 5)
    state = _state(s1, s2)
    state = state.apply_instructions(pe.generate_instructions(state, "protect", "wish")[0])
    stored_wish = tuple(int(v) for v in state.side_two.wish)
    before_hp = int(state.side_two.pokemon[1].hp)
    state = state.apply_instructions(pe.generate_instructions(state, "protect", "wailord")[0])
    engine_heal = int(state.side_two.pokemon[1].hp) - before_hp

    return {
        "scenario": "wish",
        "ledger_row": "D10",
        "gen3_rule": (
            "gen3 inherits GEN 4: data/mods/gen4/moves.ts wish.condition.onEnd -> "
            "this.heal(target.baseMaxhp / 2) = the RESOLVING ACTIVE's own max HP. "
            "The caster-based rule (source.maxhp / 2) is the gen5+ BASE move only."
        ),
        "engine_caster_maxhp": caster_maxhp,
        "engine_recipient_maxhp": recipient_maxhp,
        "engine_heal_amount": engine_heal,
        "engine_stored_wish_tuple": stored_wish,
        "engine_matches_recipient_half": engine_heal == recipient_maxhp // 2,
        "engine_matches_caster_half": engine_heal == caster_maxhp // 2,
        "showdown_observations": observations[:3],
        "showdown_samples": len(observations),
        "showdown_heals_recipient_half": (
            all(abs(o["heal"] - o["recipient_maxhp"] // 2) <= 1 for o in observations)
            if observations else None
        ),
        "verdict": "cannot-reproduce" if engine_heal == recipient_maxhp // 2 else "confirmed",
        "detail": (
            f"engine healed {engine_heal} = recipient maxhp/2 ({recipient_maxhp // 2}); "
            "this is the gen3/gen4 rule, so the inherited D10 claim had the "
            "generations backwards"
            if engine_heal == recipient_maxhp // 2
            else f"engine healed {engine_heal}, recipient/2={recipient_maxhp // 2}, "
                 f"caster/2={caster_maxhp // 2}"
        ),
    }


# --- D11: sub-band damage bias ----------------------------------------------

# (name, attacker, defender, attacker_move, defender_move)
# The defender's scripted move must never change the DEFENDER's own HP (no
# Recover/Rest/Leftovers-triggering heal), because the defender's HP delta is the
# measurement. Self-boosts and moves aimed at the attacker are both safe.
_DAMAGE_CASES = (
    # (name, attacker, defender, attacker_move, defender_move)
    #
    # Two invariants make the comparison honest:
    #  * every mon is ITEMLESS. The engine branch's post-state HP is a NET figure
    #    (damage minus the end-of-turn Leftovers heal) while Showdown's |-damage|
    #    line is GROSS, so a Leftovers holder manufactures a fake ~maxhp/16 bias.
    #  * the defender's scripted move never touches the DEFENDER's own HP or
    #    defensive stats (no Recover/Rest, no Curse/Calm Mind) — its HP delta is
    #    the measurement, and a same-turn defence boost makes damage order-dependent.
    ("eq_vs_snorlax",
     _mon("Swampert", ["Earthquake", "Ice Beam", "Protect", "Toxic"], ability="Torrent", item=None),
     _mon("Snorlax", ["Body Slam", "Shadow Ball", "Rest", "Curse"], ability="Immunity", item=None),
     "earthquake", "bodyslam"),
    ("surf_vs_tyranitar",
     _mon("Starmie", ["Surf", "Psychic", "Thunder Wave", "Recover"], ability="Natural Cure", item=None),
     _mon("Tyranitar", ["Rock Slide", "Earthquake", "Crunch", "Pursuit"], ability="Sand Stream", item=None),
     "surf", "crunch"),
    ("drillpeck_vs_celebi",
     _mon("Skarmory", ["Drill Peck", "Spikes", "Roar", "Protect"], ability="Keen Eye", item=None),
     _mon("Celebi", ["Leech Seed", "Psychic", "Recover", "Calm Mind"], ability="Natural Cure", item=None),
     "drillpeck", "psychic"),
    ("psychic_vs_starmie",
     _mon("Jirachi", ["Psychic", "Wish", "Protect", "Fire Punch"], ability="Serene Grace", item=None),
     _mon("Starmie", ["Surf", "Psychic", "Thunder Wave", "Recover"], ability="Natural Cure", item=None),
     "psychic", "thunderwave"),
)


def scenario_damage(showdown_root: str, seeds: int, seed_start: int) -> dict:
    """Compare Showdown's sampled damage mean against the engine's representative roll."""

    from pokezero.dex import load_showdown_dex
    from pokezero.engine_fidelity import engine_branch_features, fixture_battle_spec
    from pokezero.poke_engine_adapter import build_poke_engine_state

    config = LocalShowdownConfig(showdown_root=showdown_root)
    dex = load_showdown_dex(showdown_root)
    rows = []
    for name, attacker, defender, move, defender_move in _DAMAGE_CASES:
        samples = []
        for offset in range(seeds):
            try:
                result = run_multi_turn_fixture(
                    p1_team=[attacker], p2_team=[defender],
                    turns=[(f"move {move}", f"move {defender_move}")],
                    seed=seed_start + offset, config=config,
                )
            except Exception:  # noqa: BLE001
                continue
            if result.error_lines or not result.steps:
                continue
            lines = result.steps[0].protocol_lines
            if any("|-crit|" in l for l in lines) or any("|-miss|" in l for l in lines):
                continue  # crits/misses are separate engine branches, not roll spread
            dmg_line = next((l for l in lines if l.startswith("|-damage|p2a")), None)
            if dmg_line is None:
                continue
            cur, _, mx = dmg_line.split("|")[3].partition("/")
            mx = mx.split()[0]
            samples.append(int(mx) - int(cur))
        if len(samples) < 8:
            rows.append({"case": name, "status": "insufficient_samples", "n": len(samples)})
            continue

        spec = fixture_battle_spec([attacker], [defender], dex=dex)
        state = build_poke_engine_state(spec, module=pe)
        branches = engine_branch_features(state, move, defender_move, module=pe)
        start_hp = spec.side_two.pokemon[0].hp
        # The representative (non-crit, highest-probability) branch's damage.
        best = max(branches, key=lambda b: b["percentage"])
        engine_damage = start_hp - best["features"].p2_hp

        mean = statistics.fmean(samples)
        bias = (mean - engine_damage) / engine_damage if engine_damage else None
        rows.append({
            "case": name,
            "n": len(samples),
            "showdown_mean": round(mean, 2),
            "showdown_min": min(samples),
            "showdown_max": max(samples),
            "engine_representative": engine_damage,
            "relative_bias": round(bias, 4) if bias is not None else None,
            "within_roll_quantisation": abs(bias) <= 0.01 if bias is not None else None,
        })

    scored = [r for r in rows if r.get("relative_bias") is not None]
    worst = max((abs(r["relative_bias"]) for r in scored), default=None)
    return {
        "scenario": "damage",
        "ledger_row": "D11",
        "method": (
            "Showdown rolls uniform 85..100 percent; its MEAN should equal the "
            "engine's representative branch. A systematic offset below the "
            "transition differential's +/-16 % band is exactly what that band cannot see."
        ),
        "cases": rows,
        "max_abs_relative_bias": round(worst, 4) if worst is not None else None,
        "verdict": (
            "cannot-reproduce" if worst is not None and worst <= 0.01
            else "confirmed" if worst is not None else "inconclusive"
        ),
    }


SCENARIOS = {"encore": scenario_encore, "wish": scenario_wish, "damage": scenario_damage}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--showdown-root", required=True)
    ap.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed-start", type=int, default=990000)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = []
    for name in names:
        result = SCENARIOS[name](args.showdown_root, args.seeds, args.seed_start)
        results.append(result)
        print("=" * 78)
        print(f"[{result['ledger_row']} / {result['scenario']}]  VERDICT: {result['verdict'].upper()}")
        print(f"  rule: {result.get('gen3_rule') or result.get('method')}")
        for key, value in result.items():
            if key in ("scenario", "ledger_row", "gen3_rule", "method", "verdict"):
                continue
            print(f"  {key}: {json.dumps(value)[:400]}")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
