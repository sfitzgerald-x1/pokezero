#!/usr/bin/env python
"""Showdown-vs-engine differential for the gen3 Rapid Spin / Protect fidelity patch.

This is the residual/attract-caliber ground-truth gate for
``third_party/poke-engine-gen3-rapidspin-fidelity.patch``. It drives curated gen3
Custom Game scenarios through BOTH:

  * the real Node Showdown sim (``pokezero.showdown_fixture`` -> ``battle_bridge.mjs``),
    reading the measured (Rapid Spin) turn's protocol to decide what happened, and
  * the patched ``poke_engine`` (``generate_instructions``), reading the exact
    instruction list for the SAME interaction (identical conditions present),

and asserts both agree with the hand-verified gen3 ground truth:

  protect      : Rapid Spin into Protect -> hazards STAY (bug: engine used to strip
                 them because move_id survives remove_effects_for_protect()).
  substitute   : Rapid Spin into a Substitute -> hazards CLEARED (the spin connected
                 on the sub; the guard is on Protect, NOT on hit_sub).
  ghost        : Rapid Spin into a Ghost (Normal-immune) -> hazards STAY (engine
                 already correct via type immunity; this is the regression guard).
  connecting   : Rapid Spin connecting normally -> hazards CLEARED.
  leechseed    : a Leech-Seeded spinner -> Leech Seed ENDS (newly modelled).
  partialtrap  : a Wrapped/Fire-Spun spinner -> partial-trap ENDS (newly modelled).
  siblingprotect: Seismic Toss into Protect -> NO damage (sibling sweep: the
                 move-id-keyed special effect no longer fires through Protect).

Each scenario runs identical conditions on the sim and the engine. The spin effect
is deterministic; the only randomness is whether a 90%/85%-accurate SETUP move
(Leech Seed, Fire Spin) lands, so those scenarios only assert on seeds where the
setup actually landed and require at least one such seed.

MUST run in the dedicated rapidspin venv (never the shared one):
    .venv-rapidspin/bin/python scripts/rapidspin_differential.py \
        --showdown-root /Users/scott/workspace/pokerena/vendor/pokemon-showdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokezero.local_showdown import LocalShowdownConfig
from pokezero.showdown_fixture import FixturePokemon, run_multi_turn_fixture

try:
    import poke_engine
except ImportError:  # pragma: no cover
    poke_engine = None


# --- curated gen3 Custom Game sets ------------------------------------------

def _forretress():  # spinner (p2)
    return FixturePokemon(species="Forretress", ability="Sturdy", item="Leftovers",
                          moves=("Rapid Spin", "Protect", "Toxic", "Spikes"))

def _machamp():  # sibling-sweep attacker (p2): Seismic Toss is keyed in choice_special_effect
    return FixturePokemon(species="Machamp", ability="Guts", item="Leftovers",
                          moves=("Seismic Toss", "Super Fang", "Protect", "Toxic"))

def _skarmory():  # hazard setter + protecter (p1)
    return FixturePokemon(species="Skarmory", ability="Keen Eye", item="Leftovers",
                          moves=("Spikes", "Protect", "Toxic", "Roar"))

def _blissey_sub():  # substitute user (p1)
    return FixturePokemon(species="Blissey", ability="Natural Cure", item="Leftovers",
                          moves=("Substitute", "Soft-Boiled", "Toxic", "Seismic Toss"))

def _gengar():  # ghost (p1)
    return FixturePokemon(species="Gengar", ability="Levitate", item="Leftovers",
                          moves=("Substitute", "Night Shade", "Toxic", "Will-O-Wisp"))

def _cacturne():  # leech seed setter (p1)
    return FixturePokemon(species="Cacturne", ability="Sand Veil", item="Leftovers",
                          moves=("Leech Seed", "Spikes", "Toxic", "Substitute"))

def _ninetales():  # partial-trap setter (p1)
    return FixturePokemon(species="Ninetales", ability="Flash Fire", item="Leftovers",
                          moves=("Fire Spin", "Toxic", "Roar", "Will-O-Wisp"))


def _has(lines, needle: str) -> bool:
    return any(needle in l for l in lines)


# --- scenario specs ---------------------------------------------------------
# Each spec is identical on both engines. `expect` is the ground truth for the
# clears; `setup_step`/`setup_landed` gate the probabilistic-setup scenarios.

def _spec(name):
    if name == "protect":
        return dict(
            p1=[_skarmory()], p2=[_forretress()],
            turns=[("move spikes", "move toxic"), ("move protect", "move rapidspin")],
            measured=1, setup_step=None, setup_landed=None,
            expect={"spikes": False, "leech": False, "trap": False},
            landmark=lambda L: _has(L, "|-activate|") and _has(L, "Protect"),
            landmark_desc="Protect activated")
    if name == "connecting":
        return dict(
            p1=[_skarmory()], p2=[_forretress()],
            turns=[("move spikes", "move toxic"), ("move toxic", "move rapidspin")],
            measured=1, setup_step=None, setup_landed=None,
            expect={"spikes": True, "leech": False, "trap": False},
            landmark=lambda L: True, landmark_desc="")
    if name == "substitute":
        return dict(
            p1=[_skarmory(), _blissey_sub()], p2=[_forretress()],
            turns=[("move spikes", "move toxic"), ("switch 2", "move toxic"),
                   ("move substitute", "move toxic"), ("move softboiled", "move rapidspin")],
            measured=3, setup_step=None, setup_landed=None,
            expect={"spikes": True, "leech": False, "trap": False},
            landmark=lambda L: _has(L, "|-activate|") and _has(L, "Substitute"),
            landmark_desc="Substitute activated")
    if name == "ghost":
        return dict(
            p1=[_skarmory(), _gengar()], p2=[_forretress()],
            turns=[("move spikes", "move toxic"), ("switch 2", "move toxic"),
                   ("move nightshade", "move rapidspin")],
            measured=2, setup_step=None, setup_landed=None,
            expect={"spikes": False, "leech": False, "trap": False},
            landmark=lambda L: _has(L, "|-immune|"), landmark_desc="Ghost immune")
    if name == "leechseed":
        return dict(
            p1=[_cacturne()], p2=[_forretress()],
            turns=[("move leechseed", "move toxic"), ("move toxic", "move rapidspin")],
            measured=1, setup_step=0,
            setup_landed=lambda L: _has(L, "|-start|") and _has(L, "move: Leech Seed")
                                   and not _has(L, "[miss]"),
            expect={"spikes": False, "leech": True, "trap": False},
            landmark=lambda L: True, landmark_desc="")
    if name == "partialtrap":
        return dict(
            p1=[_ninetales()], p2=[_forretress()],
            turns=[("move firespin", "move toxic"), ("move toxic", "move rapidspin")],
            measured=1, setup_step=0,
            setup_landed=lambda L: _has(L, "|-activate|") and _has(L, "move: Fire Spin")
                                   and not _has(L, "[miss]"),
            expect={"spikes": False, "leech": False, "trap": True},
            landmark=lambda L: True, landmark_desc="")
    if name == "siblingprotect":
        return dict(
            p1=[_skarmory()], p2=[_machamp()],
            turns=[("move protect", "move seismictoss")],
            measured=0, setup_step=None, setup_landed=None,
            expect={"sibling_damage": False},
            landmark=lambda L: _has(L, "|-activate|") and _has(L, "Protect"),
            landmark_desc="Protect activated")
    raise ValueError(name)


def sim_facts(lines) -> dict:
    return {
        "spikes": _has(lines, "|-sideend|") and _has(lines, "Spikes")
                  and _has(lines, "move: Rapid Spin"),
        "leech": _has(lines, "|-end|") and _has(lines, "Leech Seed")
                 and _has(lines, "move: Rapid Spin"),
        "trap": _has(lines, "|-end|") and _has(lines, "[partiallytrapped]"),
        "sibling_damage": any(l.startswith("|-damage|p1") for l in lines),
    }


# --- engine side ------------------------------------------------------------

def _emon(species, moves, types=("normal", "typeless")):
    pe = poke_engine
    return pe.Pokemon(
        id=species, level=80, types=types, hp=250, maxhp=300, ability="none",
        item="none", attack=150, defense=150, special_attack=150,
        special_defense=150, speed=100, moves=[pe.Move(id=m, pp=16) for m in moves])


def _estate(*, defender_vols=frozenset(), attacker_vols=frozenset(), spikes,
            defender_types=("normal", "typeless"), sub_health=0, attacker_move="rapidspin"):
    pe = poke_engine
    dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
    s1 = pe.Side(active_index="0",
                 pokemon=[_emon("attacker", [attacker_move, "tackle"])] + [dummy] * 5,
                 volatile_statuses=set(attacker_vols),
                 side_conditions=pe.SideConditions(spikes=spikes))
    s2 = pe.Side(active_index="0",
                 pokemon=[_emon("defender", ["splash", "tackle"], types=defender_types)] + [dummy] * 5,
                 volatile_statuses=set(defender_vols), substitute_health=sub_health)
    return pe.State(side_one=s1, side_two=s2, weather="none", terrain="none", trick_room=False)


def engine_facts(name) -> dict:
    if name == "protect":
        state, move = _estate(defender_vols={"PROTECT"}, spikes=2), "rapidspin"
    elif name == "connecting":
        state, move = _estate(spikes=2), "rapidspin"
    elif name == "substitute":
        state, move = _estate(defender_vols={"SUBSTITUTE"}, spikes=2, sub_health=80), "rapidspin"
    elif name == "ghost":
        state, move = _estate(defender_types=("ghost", "poison"), spikes=2), "rapidspin"
    elif name == "leechseed":
        state, move = _estate(attacker_vols={"LEECHSEED"}, spikes=0), "rapidspin"
    elif name == "partialtrap":
        state, move = _estate(attacker_vols={"PARTIALLYTRAPPED"}, spikes=0), "rapidspin"
    elif name == "siblingprotect":
        state, move = _estate(defender_vols={"PROTECT"}, spikes=0, attacker_move="seismictoss"), "seismictoss"
    else:
        raise ValueError(name)
    branches = poke_engine.generate_instructions(state, move, "splash")
    insts = [str(i) for b in branches for i in b.instruction_list]
    return {
        "spikes": any("ChangeSideCondition SideOne Spikes" in s for s in insts),
        "leech": any("RemoveVolatileStatus SideOne: LEECHSEED" in s for s in insts),
        "trap": any("RemoveVolatileStatus SideOne: PARTIALLYTRAPPED" in s for s in insts),
        "sibling_damage": any(("Damage SideTwo" in s or "DamageSubstitute SideTwo" in s) for s in insts),
    }


SCENARIOS = ["protect", "connecting", "substitute", "ghost", "leechseed", "partialtrap", "siblingprotect"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--showdown-root", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=880000)
    args = ap.parse_args(argv)

    if poke_engine is None:
        print("FAIL: poke_engine not importable (build the patched wheel into this venv)")
        return 2

    config = LocalShowdownConfig(showdown_root=args.showdown_root)
    failures: list[str] = []

    print("=" * 78)
    print("gen3 Rapid Spin / Protect fidelity differential  (Showdown vs patched engine)")
    print("=" * 78)

    for name in SCENARIOS:
        spec = _spec(name)
        exp = spec["expect"]
        eng = engine_facts(name)

        # --- engine side (deterministic) ---
        for key, want in exp.items():
            if eng[key] != want:
                failures.append(f"[{name}] engine {key}={eng[key]}, expected {want}")

        # --- sim side ---
        landed = 0
        sim_show = None
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            res = run_multi_turn_fixture(p1_team=spec["p1"], p2_team=spec["p2"],
                                         turns=spec["turns"], seed=seed, config=config)
            if len(res.steps) <= spec["measured"]:
                failures.append(f"[{name}] sim seed {seed}: no measured turn")
                continue
            if spec["setup_step"] is not None:
                setup_lines = res.steps[spec["setup_step"]].protocol_lines
                if not spec["setup_landed"](setup_lines):
                    continue  # probabilistic setup missed this seed; skip
            landed += 1
            lines = res.steps[spec["measured"]].protocol_lines
            facts = sim_facts(lines)
            sim_show = facts
            for key, want in exp.items():
                if facts[key] != want:
                    failures.append(f"[{name}] sim seed {seed}: {key}={facts[key]}, expected {want}")
            if not spec["landmark"](lines):
                failures.append(f"[{name}] sim seed {seed}: missing landmark ({spec['landmark_desc']})")

        need = 1 if spec["setup_step"] is not None else args.seeds
        if landed < need:
            failures.append(f"[{name}] only {landed} usable sim seeds (need >= {need})")

        print(f"\n[{name}]  usable_sim_seeds={landed}/{args.seeds}")
        print(f"    engine: spikes={eng['spikes']} leech={eng['leech']} trap={eng['trap']} "
              f"sibling_damage={eng['sibling_damage']}")
        if sim_show is not None:
            print(f"    sim   : spikes={sim_show['spikes']} leech={sim_show['leech']} "
                  f"trap={sim_show['trap']} sibling_damage={sim_show['sibling_damage']}")

    print("\n" + "=" * 78)
    if failures:
        print("DIFFERENTIAL FAILED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("DIFFERENTIAL PASSED: Showdown ground truth matches the patched engine on all "
          "Protect / Substitute / Ghost / connecting / Leech-Seed / partial-trap / sibling cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
