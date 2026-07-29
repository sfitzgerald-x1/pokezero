#!/usr/bin/env python
"""Standing behavioral probes for the vendored gen3 engine build. RUN AFTER
EVERY WHEEL REBUILD, before believing any engine measurement.

Why this exists as a committed script: ``engine_build_fingerprint.py --check``
verifies the *patch set* against the *stamp*, but a vendor re-run that restamps
without rebuilding the wheel leaves a stale ``.so`` that --check calls current
(ledger Appendix D.4 / review finding F1 on PR #941 — it has bitten twice).
Behavior is the only thing a stale wheel cannot fake. Cycle eight recorded the
two stepwise discriminators "NOT RUN — fixture not sourceable in this checkout"
(ledger Appendix Z2.4): the right refusal, fixed here by committing the battery
fixtures themselves.

PROVENANCE OF EXPECTED VALUES — transcribed, not re-derived. Every fixture and
every expected constant below is copied from the PR #955 battery agent's
artifacts (its ``probe_battery.py`` and ``struggle_probe.log``), which measured
them against the 33-patch build fingerprint
887a722dd2d6cd9b16c7e9736e07f0f5e7f591b17e38a8b9a7a593f31bc6659d and
sim-anchored the Struggle relation against the vendored Showdown. The
sim-anchor log line, quoted verbatim so this file carries its own provenance:

    RELATION healthy_noncrit_max=194 burned_noncrit_max=98 stepwise_expect=98
    old_engine_expect=97 verdict=STEPWISE

(That line is the Golem-vs-Blissey forced-Struggle sim probe: all 16 rolls
observed in both arms; the old pipeline's 97 was UNREACHABLE in the sim.)

The four probes and what each discriminates:

  1. PAIN SPLIT CLAMP (patch 29): Weezing 238/238 vs Groudon 252/252 ->
     ``Damage SideOne: 0`` / ``Damage SideTwo: 7``. An unpatched engine emits a
     symmetric split with no maxhp clamp (ledger Appendix S.5).
  2. PROTECT STALL LADDER (patch 30): success mass at consecutive-success
     count k = 0..4 -> 100 / 50 / 25 / 12.5 / 12.5. The k=4 value pins the
     gen3 ``stall`` counterMax=8 floor (1/8 from the 5th attempt; crate test
     ``the_fifth_attempt_holds_at_one_eighth`` in
     tests/gen3_protect_floor_fidelity.rs is the Rust-level twin). An
     unpatched engine keeps halving: 6.25 at k=4.
  3. STAB-ODD STEPWISE DISCRIMINATOR (patch 33): STAB Surf, odd (B+2), into a
     2x-weak defender. The sim floors after EVERY modifier step
     (gen3 ``modifyDamage``); the engine used to multiply STAB x type in f32
     and floor once, letting the STAB half-point survive the x2 type step.
     Battery fixture (spa=230, def=200, L100 => B=91): stepwise non-crit max
     ** 278 **; old float pipeline ** 279 **. One integer, one call, decides
     the pipeline.
  4. BURNED-STRUGGLE ORDER DISCRIMINATOR (patch 33): burn halves BEFORE the
     +2 and the crit x2 (stepwise), not after everything (old). Battery
     fixture (atk=180 vs def=140, L100, crit): healthy crit max ** 112 **,
     burned crit max ** 58 ** stepwise vs ** 56 ** old. Sim-anchored by the
     quoted RELATION line above.

Symbols involved (content-addressed deliberately — cite symbols, never line
numbers; a "line 2495" citation went stale mid-review when #955's rewrite moved
the region): the branch-applied damage is ``avg_damage_dealt`` =
trunc(0.925 * max) in ``gen3/generate_instructions.rs``; the probe-visible
rolls come from ``calculate_damage(..)`` / ``DamageRolls::Max``; the stepwise
pipeline lives in ``common_pkmn_damage_calc`` in ``gen3/damage_calc.rs``.

Usage::

    .venv/bin/python scripts/engine_behavioral_probes.py

Exit code 0 = all probes match the transcribed expectations (build is
behaviorally current at 33 patches). Nonzero = DO NOT trust engine
measurements from this venv; rebuild the wheel and re-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import poke_engine as pe

REPO_ROOT = Path(__file__).resolve().parents[1]

FAILURES: list[str] = []


def _report(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{name}] {tag} {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def _dummy() -> pe.Pokemon:
    return pe.Pokemon(id="pikachu", level=1, hp=0)


def _mk_state(p1: pe.Pokemon, p2: pe.Pokemon, p1_conditions=None) -> pe.State:
    kw1 = {}
    if p1_conditions is not None:
        kw1["side_conditions"] = p1_conditions
    return pe.State(
        side_one=pe.Side(active_index="0", pokemon=[p1] + [_dummy()] * 5, **kw1),
        side_two=pe.Side(active_index="0", pokemon=[p2] + [_dummy()] * 5),
        weather="none",
        terrain="none",
        trick_room=False,
    )


# ---------------------------------------------------------------------------
# Probe 1: Pain Split clamp (patch 29).
# Fixture transcribed from the #955 battery's probe_battery.py.
# Expected: Damage SideOne: 0 / Damage SideTwo: 7 (ledger S.5, U.1, Z2.4).
# ---------------------------------------------------------------------------
def probe_painsplit() -> None:
    weezing = pe.Pokemon(
        id="weezing", level=100, types=("poison", "typeless"),
        hp=238, maxhp=238, ability="levitate", item="none",
        attack=200, defense=240, special_attack=200,
        special_defense=170, speed=140,
        moves=[pe.Move(id="painsplit", pp=16)],
    )
    groudon = pe.Pokemon(
        id="groudon", level=100, types=("ground", "typeless"),
        hp=252, maxhp=252, ability="drought", item="none",
        attack=300, defense=280, special_attack=200,
        special_defense=180, speed=180,
        moves=[pe.Move(id="splash", pp=16)],
    )
    state = _mk_state(weezing, groudon)
    branches = pe.generate_instructions(state, "painsplit", "splash")
    lines = [str(i) for b in branches for i in b.instruction_list]
    damage = [l for l in lines if l.startswith("Damage")]
    expected = ["Damage SideOne: 0", "Damage SideTwo: 7"]
    _report(
        "painsplit-clamp",
        damage == expected,
        f"expected {expected}, got {damage}",
    )


# ---------------------------------------------------------------------------
# Probe 2: Protect stall ladder k=0..4 (patch 30).
# Fixture transcribed from the #955 battery's probe_battery.py (k=0..3);
# the k=4 floor expectation (12.5, not 6.25) is the patch-30 counterMax=8
# floor, sim-measured in ledger Appendix L/#944 and pinned in the crate test
# the_fifth_attempt_holds_at_one_eighth.
# ---------------------------------------------------------------------------
def probe_protect_ladder() -> None:
    expected = {0: 100.0, 1: 50.0, 2: 25.0, 3: 12.5, 4: 12.5}
    for k, want in expected.items():
        prot = pe.Pokemon(
            id="spinda", level=100, types=("normal", "typeless"),
            hp=282, maxhp=282, ability="owntempo", item="none",
            attack=180, defense=180, special_attack=180,
            special_defense=180, speed=180,
            moves=[pe.Move(id="protect", pp=16)],
        )
        atk = pe.Pokemon(
            id="porygon2", level=100, types=("normal", "typeless"),
            hp=310, maxhp=310, ability="trace", item="none",
            attack=220, defense=200, special_attack=230,
            special_defense=210, speed=150,
            moves=[pe.Move(id="return102", pp=16)],
        )
        state = _mk_state(prot, atk, p1_conditions=pe.SideConditions(protect=k))
        branches = pe.generate_instructions(state, "protect", "return102")
        mass = 0.0
        for b in branches:
            ins = [str(i) for i in b.instruction_list]
            if not any(l.startswith("Damage SideOne:") for l in ins):
                mass += b.percentage
        _report(
            f"protect-ladder-k{k}",
            abs(mass - want) < 1e-6,
            f"expected {want}, got {mass:.4f}",
        )


# ---------------------------------------------------------------------------
# Probe 3: STAB-odd stepwise discriminator (patch 33).
# Fixture transcribed from the #955 battery's probe_battery.py: Starmie
# (water, spa=230) STAB Surf into Charizard (fire, spd=200), both L100.
# B = tr(tr(tr(2*100/5+2)*95*230 / 200) / 50) = 91; B+2 = 93 odd.
# Expected non-crit max: stepwise floor(1.5*93)*2 = 278; old float 3*93 = 279.
# ---------------------------------------------------------------------------
def probe_stepwise_stab() -> None:
    attacker = pe.Pokemon(
        id="starmie", level=100, types=("water", "typeless"),
        hp=300, maxhp=300, ability="illuminate", item="none",
        attack=150, defense=180, special_attack=230,
        special_defense=190, speed=250,
        moves=[pe.Move(id="surf", pp=16)],
    )
    defender = pe.Pokemon(
        id="charizard", level=100, types=("fire", "typeless"),
        hp=300, maxhp=300, ability="blaze", item="none",
        attack=200, defense=190, special_attack=220,
        special_defense=200, speed=200,
        moves=[pe.Move(id="splash", pp=16)],
    )
    state = _mk_state(attacker, defender)
    rolls, _ = pe.calculate_damage(state, "surf", "splash", False)
    got = rolls[0]
    _report(
        "stepwise-stab-odd",
        got == 278,
        f"expected 278 (stepwise; old float pipeline gives 279), got {got}",
    )


# ---------------------------------------------------------------------------
# Probe 4: burned-Struggle order discriminator (patch 33).
# Fixture transcribed from the #955 battery's probe_battery.py: L100
# attacker atk=180 vs defender def=140, Struggle, crit. B = 54.
# Healthy crit max 2*(B+2) = 112. Burned crit max: stepwise halves BEFORE
# the +2 and the crit x2 -> 2*(floor(B/2)+2) = 58; old trailing halving
# -> floor((2*(B+2))/2) = 56. Sim-anchored: see the quoted RELATION line in
# the module docstring (verdict=STEPWISE; old value unreachable in sim).
# ---------------------------------------------------------------------------
def probe_burned_struggle() -> None:
    def struggle_crit_max(status: str) -> int:
        attacker = pe.Pokemon(
            id="attacker", level=100,
            types=("normal", "typeless"), base_types=("normal", "typeless"),
            hp=350, maxhp=350, ability="none", item="none",
            attack=180, defense=140, special_attack=170,
            special_defense=130, speed=120, status=status,
        )
        defender = pe.Pokemon(
            id="defender", level=100,
            types=("water", "typeless"), base_types=("water", "typeless"),
            hp=350, maxhp=350, ability="none", item="none",
            attack=180, defense=140, special_attack=170,
            special_defense=130, speed=120,
        )
        state = _mk_state(attacker, defender)
        rolls, _ = pe.calculate_damage(state, "struggle", "splash", True)
        return rolls[1]

    healthy = struggle_crit_max("none")
    burned = struggle_crit_max("burn")
    _report(
        "burned-struggle-healthy",
        healthy == 112,
        f"expected healthy crit max 112, got {healthy}",
    )
    _report(
        "burned-struggle-crit",
        burned == 58,
        f"expected 58 (stepwise; old trailing halving gives 56), got {burned}",
    )


def _print_build_identity() -> None:
    stamp = Path(sys.prefix) / ".engine-build-fingerprint.json"
    if stamp.exists():
        data = json.loads(stamp.read_text())
        print(
            f"[build] fingerprint {data.get('fingerprint', '?')} "
            f"({len(data.get('patches', []))} patches)"
        )
    else:
        print(f"[build] WARNING: no stamp at {stamp} — run "
              "scripts/engine_build_fingerprint.py --write after building")


def main() -> int:
    _print_build_identity()
    probe_painsplit()
    probe_protect_ladder()
    probe_stepwise_stab()
    probe_burned_struggle()
    if FAILURES:
        print(f"\n{len(FAILURES)} probe(s) FAILED — the installed wheel does "
              "not behave like the 33-patch engine. Rebuild before measuring.")
        return 1
    print("\nall behavioral probes PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
