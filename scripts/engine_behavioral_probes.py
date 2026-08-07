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
     burned crit max ** 58 ** stepwise vs ** 56 ** old. Evidence labels,
     kept separate: the quoted RELATION line sim-anchors the NON-CRIT
     relation only (all 16 rolls observed both arms; old value 97
     unreachable). The CRIT expectations here are derived from the stepwise
     model applied to this fixture, NOT from sim measurement — the sim's
     crit sample was too small to discriminate (its CRIT RELATION line:
     healthy_crit_max=384 burned_crit_max=192 pin_expect=194; 8 crits, 6
     distinct values, under-sampled max, no verdict). The crit probe is a
     valid REGRESSION pin (a revert to trailing halving flips 58 -> 56),
     not a sim-confirmed discriminator.

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
# -> floor((2*(B+2))/2) = 56.
# Evidence label: the sim anchor (quoted RELATION line, module docstring)
# covers the NON-CRIT relation only. These crit expectations are derived
# from the stepwise model applied to this fixture; the sim's crit sample
# was too small to discriminate (8 crits / 6 distinct values, max 192 vs
# stepwise pin 194 — under-sampled, no verdict). Valid regression pin for
# a revert to trailing halving; not a sim-confirmed discriminator.
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



# ---------------------------------------------------------------------------
# Probe 5: Gen 3 contact flags (patch 58).
# Gen 3 assigns contact differently from Gen 4+: Overheat and Ancient Power
# make contact, while Covet, Fake Out and Feint Attack do not. Upstream
# poke-engine carries the Gen 4+ flags. Rough Skin is the observable: it only
# retaliates on contact, so the flag shows up as a recoil Damage instruction
# against the attacker.
# ---------------------------------------------------------------------------
def probe_contact_flags() -> None:
    def attacker_takes_rough_skin(move: str) -> bool:
        attacker = pe.Pokemon(
            id="attacker", level=100,
            types=("normal", "typeless"), base_types=("normal", "typeless"),
            hp=320, maxhp=320, ability="none", item="none",
            attack=200, defense=180, special_attack=200,
            special_defense=170, speed=200,
            moves=[pe.Move(id=move, pp=16)],
        )
        defender = pe.Pokemon(
            id="defender", level=100,
            types=("water", "typeless"), base_types=("water", "typeless"),
            hp=320, maxhp=320, ability="roughskin", item="none",
            attack=180, defense=180, special_attack=170,
            special_defense=170, speed=100,
            moves=[pe.Move(id="splash", pp=16)],
        )
        state = _mk_state(attacker, defender)
        branches = pe.generate_instructions(state, move, "splash")
        lines = [str(i) for b in branches for i in b.instruction_list]
        # Rough Skin costs the attacker maxhp/16 == 20.
        return any(l == "Damage SideOne: 20" for l in lines)

    for move, want_contact in (
        ("overheat", True),
        ("ancientpower", True),
        ("covet", False),
        ("fakeout", False),
        ("feintattack", False),
    ):
        got = attacker_takes_rough_skin(move)
        _report(
            f"gen3-contact-{move}",
            got == want_contact,
            f"expected contact={want_contact} (rough skin recoil), got {got}",
        )



# ---------------------------------------------------------------------------
# Probe 6: residual-lethality partition, incl. the Toxic ladder (patch 57).
# A damage roll can decide lethality one phase later. The branch that cannot
# kill on the hit must still split on `hp - pending_residual_damage`.
#
# The Toxic rung is the sharp case: add_end_of_turn_instructions computes
# `stage = normalized_toxic_count + 1`, so a counter of 1 ticks maxhp/8, not
# maxhp/16. Mirroring it as `max(count, 1)` coincides only at count 0 and
# puts the threshold a full stage too high for every rung above it, handing
# the surviving arm rolls that in fact die.
#
# Fixture: 123/238 defender, max non-crit roll 122 (so the fan cannot kill on
# the hit), Rock Slide at 90% accuracy.
#   count 0 -> tick 14, threshold 123-14 = 109. min roll 103 < 109 <= max 122,
#              so the fan straddles the threshold: SPLIT (4 branches).
#   count 1 -> tick 28, threshold 123-28 =  95. min roll 103 > 95, so EVERY
#              non-crit roll is residual-lethal: NO split (3 branches).
# Under the old `max(count, 1)` mirror both cases split at 109, which is the
# regression this pins.
# ---------------------------------------------------------------------------
def probe_residual_lethality_partition() -> None:
    def state_for(toxic_count: int):
        attacker = pe.Pokemon(
            id="gligar", level=81,
            types=("ground", "flying"), base_types=("ground", "flying"),
            hp=205, maxhp=205, ability="none", item="none",
            attack=170, defense=160, special_attack=120,
            special_defense=130, speed=150,
            moves=[pe.Move(id="rockslide", pp=16)],
        )
        defender = pe.Pokemon(
            id="fearow", level=81,
            types=("normal", "flying"), base_types=("normal", "flying"),
            hp=123, maxhp=238, ability="none", item="none",
            attack=170, defense=145, special_attack=110,
            special_defense=125, speed=180, status="toxic",
            moves=[pe.Move(id="splash", pp=16)],
        )
        state = pe.State(
            side_one=pe.Side(active_index="0", pokemon=[attacker] + [_dummy()] * 5),
            side_two=pe.Side(
                active_index="0", pokemon=[defender] + [_dummy()] * 5,
                side_conditions=pe.SideConditions(toxic_count=toxic_count),
            ),
            weather="none", terrain="none", trick_room=False,
        )
        return state

    def branches(toxic_count: int) -> list:
        return pe.generate_instructions(
            state_for(toxic_count), "rockslide", "splash"
        )

    # The whole fixture rests on the fan being unable to kill on the hit while
    # straddling the count-0 threshold. Assert that, do not assume it: if a
    # damage-formula patch shifts the roll, every branch count below is
    # meaningless and this is the assertion that says so.
    max_regular = pe.calculate_damage(
        state_for(0), "rockslide", "splash", False
    )[0][0]
    _report(
        "residual-partition-fixture",
        max_regular == 122,
        f"fixture requires max non-crit roll 122 (< 123 hp, > threshold 109), "
        f"got {max_regular}",
    )
    straddling = branches(0)
    saturated = branches(1)
    _report(
        "residual-partition-splits-when-fan-straddles",
        len(straddling) == 4,
        f"toxic_count=0 (tick 14, threshold 109): expected 4 branches, "
        f"got {len(straddling)}",
    )
    _report(
        "residual-partition-toxic-stage-is-count-plus-one",
        len(saturated) == 3,
        f"toxic_count=1 (tick 28, threshold 95) leaves no surviving roll, so "
        f"the fan must NOT split: expected 3 branches, got {len(saturated)}. "
        f"4 means the mirror used max(count, 1) instead of count + 1.",
    )
    # The saturated case must also actually kill: 112 + clamped 11 == 123.
    hit = [b for b in saturated if 80.0 < b.percentage < 90.0]
    lethal = bool(hit) and sum(
        int(str(i).split(": ")[1])
        for i in hit[0].instruction_list
        if str(i).startswith("Damage SideTwo")
    ) == 123
    _report(
        "residual-partition-saturated-arm-is-lethal",
        lethal,
        "the single non-crit arm must sum to exactly the defender's 123 hp",
    )



# ---------------------------------------------------------------------------
# Probe 7: residual-partition branch MASSES (patch 57).
#
# Branch counts and damage sums cannot see a probability-mass leak, and neither
# can the transition differential -- it compares roll-scaled components, not
# masses, so a leak measures as "neutral" on a 200-game sweep. This probe
# compares the engine's total KO mass against an independent reconstruction:
# enumerate the 16 gen3 rolls as floor(max * r / 100) for r in 85..=100, read
# the true residual tick from a turn where neither side attacks, and count the
# rolls that die. No reference to the partition's own arithmetic.
#
# The regression this pins is real and was caught in review: the non-crit
# residual split used to call `incoming_instructions.update_percentage` in
# place, but every crit arm below clones that same value, so the crit arms were
# silently scaled by the non-crit factor and the stolen mass landed on the
# non-crit survive arm. Totals still summed to 100%, so no conservation check
# could see it.
#
# NOTE on probe design: an HP-reading move (flail, bellydrum, painsplit) would
# force the engine's own 16-roll fan and look like a cleaner oracle, but every
# such move perturbs HP by construction, and the two confounds push in OPPOSITE
# directions:
#   * Belly Drum's self-halving is emitted as damage on the defender's own side,
#     so it INFLATES any metric summing damage to the defender.
#   * a flail KO of the attacker ends the battle, so
#     `stop_residuals_if_battle_ended` removes the defender's own tick from the
#     instruction list -- the branch then scores as a survival and the metric
#     DEFLATES.
# An earlier version of this comment said both inflate; that was wrong for
# flail, and the sign matters when reading such a probe's output.
#
# ALSO KNOWN, and not fixed here: `compare_health_with_damage_multiples`
# accumulates its roll ladder in f32, so for 174 of the first 400 `max` values
# the top rung lands one below `max`. Kill counts therefore differ from the true
# `floor(max * r / 100)` fan for exactly the positions where
# `threshold == max_damage`, where the ladder counts 0 kills and the truth counts
# 1. Repro: hp 185/320 poisoned -> tick 40, threshold 145 == max 145; truth
# 12.1094%, engine 6.2500% with no split at all. This predates the partition and
# is inherited from an identity three pre-existing paths share (Case A has the
# analogue at `threshold == hp == max_damage`), so it is registered rather than
# fixed here. None of the cases below sits on that boundary.
# ---------------------------------------------------------------------------
def probe_residual_partition_masses() -> None:
    def build(hp, maxhp, status, toxic_count, weather, attacker_move):
        attacker = pe.Pokemon(
            id="gligar", level=81,
            types=("ground", "flying"), base_types=("ground", "flying"),
            hp=205, maxhp=205, ability="none", item="none",
            attack=170, defense=160, special_attack=120,
            special_defense=130, speed=250,
            moves=[pe.Move(id=attacker_move, pp=16)],
        )
        defender = pe.Pokemon(
            id="fearow", level=81,
            types=("normal", "flying"), base_types=("normal", "flying"),
            hp=hp, maxhp=maxhp, ability="none", item="none",
            attack=170, defense=145, special_attack=110,
            special_defense=125, speed=100, status=status,
            moves=[pe.Move(id="splash", pp=16)],
        )
        kw = {}
        if toxic_count is not None:
            kw["side_conditions"] = pe.SideConditions(toxic_count=toxic_count)
        return pe.State(
            side_one=pe.Side(active_index="0", pokemon=[attacker] + [_dummy()] * 5),
            side_two=pe.Side(active_index="0", pokemon=[defender] + [_dummy()] * 5, **kw),
            weather=weather, terrain="none", trick_room=False,
        )

    def damage_to_defender(branch) -> int:
        return sum(
            int(str(i).split(": ")[1])
            for i in branch.instruction_list
            if str(i).startswith("Damage SideTwo")
        )

    accuracy, crit_rate = 0.9, 1.0 / 16.0

    def compare(label, hp, maxhp, status, toxic_count, weather):
        # True residual tick, measured with no attack in play. Two limits on
        # this oracle, both fine for the fixtures below and both worth knowing
        # before adding another: it sums only damage, so a fixture carrying
        # Leftovers or a threshold berry would net the heal against the tick and
        # mis-measure it; and `expected` below assumes the miss branch is never
        # itself a KO, which fails if hp <= tick.
        quiet = build(hp, maxhp, status, toxic_count, weather, "splash")
        tick = damage_to_defender(
            pe.generate_instructions(quiet, "splash", "splash")[0]
        )
        assert tick < hp, "fixture invalid: the residual alone would KO"
        state = build(hp, maxhp, status, toxic_count, weather, "rockslide")
        max_regular = pe.calculate_damage(state, "rockslide", "splash", False)[0][0]
        max_crit = pe.calculate_damage(state, "rockslide", "splash", True)[0][1]
        rolls = range(85, 101)
        n_regular = sum(
            1 for r in rolls if hp - (max_regular * r // 100) - tick <= 0
        )
        n_crit = sum(
            1 for r in rolls
            if hp - (max_crit * r // 100) <= 0
            or hp - (max_crit * r // 100) - tick <= 0
        )
        expected = accuracy * (
            (1.0 - crit_rate) * n_regular / 16.0 + crit_rate * n_crit / 16.0
        ) * 100.0
        actual = sum(
            b.percentage
            for b in pe.generate_instructions(state, "rockslide", "splash")
            if damage_to_defender(b) >= hp
        )
        _report(
            f"residual-mass-{label}",
            abs(actual - expected) < 0.001,
            f"tick={tick} max={max_regular}/{max_crit} rolls={n_regular}/{n_crit}: "
            f"expected KO mass {expected:.4f}%, got {actual:.4f}%",
        )

    # Non-crit fan straddles the residual threshold (the split this patch adds).
    compare("noncrit-split-toxic", 123, 238, "toxic", 0, "none")
    compare("noncrit-split-sand", 130, 320, "none", None, "sand")
    # Whole non-crit fan residual-lethal: no split, mass must not move.
    compare("saturated-toxic", 123, 238, "toxic", 1, "none")
    # NOT a Burn-tick assertion despite the status: at hp 130 the whole fan is
    # lethal anyway, so this passes identically whether or not Burn is mirrored.
    # It is a second saturated-fan case and is named for what it tests. The
    # fixture that WOULD see the missing Burn tick is hp 170/320, where the
    # truth is 70.7031% and the engine says 100.0000%; that is the open F2 gap,
    # deliberately not asserted here because it would fail today.
    compare("saturated-fan-second", 130, 320, "burn", None, "none")
    # Only the CRIT fan straddles: hp 280, tick 40 -> threshold 240, inside
    # (min_crit 207, max_crit 244], while the non-crit max of 122 cannot reach
    # it. This is the crit-fan arm the sweep left unexercised.
    compare("crit-fan-only", 280, 320, "poison", None, "none")
    # Nothing pending: the partition must not fire at all.
    compare("no-residual", 160, 320, "none", None, "none")
    # Composition with the PRE-EXISTING crit-kill split, both firing in ONE
    # call. hp 220 sits inside (min_crit 207, max_crit 244] so the crit-kill
    # split fires, and Toxic at count 4 ticks 20*5 = 100, putting the residual
    # threshold at 120 inside (min_regular 103, max_regular 122] so the non-crit
    # residual split fires too. This is the one arrangement where a second
    # in-place `update_percentage` would reintroduce the mass leak, so it must
    # not be dropped -- and both thresholds must land strictly inside their fans
    # or the case silently tests nothing.
    compare("composes-with-crit-kill-split", 220, 320, "toxic", 4, "none")



# ---------------------------------------------------------------------------
# Probe 8: the pending-residual read must see THIS call's own mutations.
#
# `pending_residual_damage` used to be bound at function level in
# `generate_instructions_from_move`, ~70 lines before
# `state.apply_instructions(&incoming_instructions.instruction_list)`. The second
# mover's incoming instructions carry the FIRST mover's entire executed action,
# so a function-level read observed the state as it was before that action --
# including before a switch. Any boundary where the defender switched in, or
# where the first mover set the weather, priced its residual threshold against a
# stale reading. reports/c111 cause A3.
#
# Both cases below are reachable from this Python harness. An earlier claim that
# they were not -- that a crate-level test using MoveChoice::Switch was required
# -- was wrong: `generate_instructions` takes a switch by the incoming Pokemon's
# ID ("fearow"), not "switch 1" / "switch fearow" / "1", which all raise
# ValueError. Found in review of PR #1065.
#
# Each pin ships with a control differing in exactly ONE variable, so neither can
# pass by accident:
#   PIN 1  defender switches in already Toxic  vs  the same Toxic mon already active
#   PIN 2  defender sets Sandstorm this turn   vs  Sandstorm already up (and vs none)
# PIN 2 needs no switch at all, so it holds even if the switch spelling regresses.
# ---------------------------------------------------------------------------
def probe_pending_read_sees_this_calls_mutations() -> None:
    def attacker() -> pe.Pokemon:
        return pe.Pokemon(
            id="gligar", level=81,
            types=("ground", "flying"), base_types=("ground", "flying"),
            hp=205, maxhp=205, ability="none", item="none",
            attack=170, defense=160, special_attack=120,
            special_defense=130, speed=250,
            moves=[pe.Move(id="rockslide", pp=16)],
        )

    def mon(pid, hp, maxhp, status, speed, move) -> pe.Pokemon:
        # FLYING matters: it makes Rock Slide 2x, which puts the NON-CRIT fan at
        # [103, 122] so the thresholds below straddle the non-crit arm -- the arm
        # that actually closed 19000058/19 and 19000198/33. With ("normal",
        # "typeless") the non-crit fan is only [51, 61], the split moves entirely
        # into the 5.625% crit arm, and a regression that reintroduced a stale
        # read at just the non-crit site would pass. Caught in review of #1065.
        return pe.Pokemon(
            id=pid, level=81,
            types=("normal", "flying"), base_types=("normal", "flying"),
            hp=hp, maxhp=maxhp, ability="none", item="none",
            attack=100, defense=145, special_attack=100,
            special_defense=125, speed=speed, status=status,
            moves=[pe.Move(id=move, pp=16)],
        )

    def branches(defender_party, defender_choice, weather):
        state = pe.State(
            side_one=pe.Side(active_index="0", pokemon=[attacker()] + [_dummy()] * 5),
            side_two=pe.Side(
                active_index="0",
                pokemon=defender_party + [_dummy()] * (6 - len(defender_party)),
                side_conditions=pe.SideConditions(toxic_count=0),
            ),
            weather=weather, terrain="none", trick_room=False,
        )
        return pe.generate_instructions(state, "rockslide", defender_choice)

    # --- PIN 1: a Toxic mon switches IN, and is the one taking the hit. -----
    # Fearow 123/238 Toxic: tick 14, threshold 109, inside the non-crit fan
    # [103, 122]. NOTE the fixture depends on max_crit (122) < hp (123) by ONE
    # point: if crit damage rises, max_crit >= hp routes to the crit-KILL split,
    # which emits two crit arms whatever `pending` is. The control catches that
    # (it would go to 4 and fail red rather than pass green), but the margin is
    # thin and deliberate. PIN 2 below has 4 HP of margin and no switch at all.
    # The control is ALSO a switch, differing only in the incoming mon's status,
    # so flinch and speed interactions are held fixed and `status` is the single
    # variable. (An earlier control let the defender use a move instead of
    # switching, which changed the branch shape for reasons unrelated to the
    # read and made the comparison meaningless.)
    outgoing = mon("slugma", 179, 250, "none", 100, "splash")
    toxic_in = branches([outgoing, mon("fearow", 123, 238, "toxic", 100, "splash")],
                        "fearow", "none")
    healthy_in = branches([outgoing, mon("fearow", 123, 238, "none", 100, "splash")],
                          "fearow", "none")
    toxic_masses = sorted(round(b.percentage, 4) for b in toxic_in)
    healthy_masses = sorted(round(b.percentage, 4) for b in healthy_in)
    _report(
        "pending-read-switched-in-defender-splits",
        len(toxic_in) == 4,
        f"a Toxic defender switching in has tick 14 and threshold 109 inside the "
        f"NON-CRIT fan [103, 122], so it must split: expected 4 branches, got "
        f"{len(toxic_in)} with masses {toxic_masses}. 3 means the read saw the "
        f"OUTGOING mon and returned 0.",
    )
    _report(
        "pending-read-switch-control-is-live",
        len(healthy_in) == 3 and toxic_masses != healthy_masses,
        f"control: switching in the SAME mon unstatused leaves nothing pending, "
        f"so it must NOT split -- expected 3 branches, got {len(healthy_in)} "
        f"({healthy_masses}). If it equals the Toxic case ({toxic_masses}) the "
        f"fixture is not measuring the residual read at all.",
    )

    # --- PIN 2: the FIRST mover sets the weather. No switch involved. -------
    # Slugma 126/250 in sand: tick 15, threshold 111, inside (103, 122].
    sand_setter = mon("slugma", 126, 250, "none", 300, "sandstorm")
    splasher = mon("slugma", 126, 250, "none", 300, "splash")
    sets_sand = sorted(round(b.percentage, 4) for b in branches([sand_setter], "sandstorm", "none"))
    sand_up = sorted(round(b.percentage, 4) for b in branches([splasher], "splash", "sand"))
    no_sand = sorted(round(b.percentage, 4) for b in branches([splasher], "splash", "none"))
    _report(
        "pending-read-weather-set-this-turn",
        sets_sand == sand_up,
        f"a defender that sets Sandstorm and outspeeds must be priced like one "
        f"already in Sandstorm: sets-this-turn {sets_sand} vs already-up "
        f"{sand_up}. Disagreement means the read predates the weather.",
    )
    _report(
        "pending-read-weather-control-is-live",
        sets_sand != no_sand,
        f"control: with no Sandstorm the threshold is not straddled and the fan "
        f"must NOT split ({no_sand}). If this equals the sand cases "
        f"({sets_sand}) the fixture proves nothing.",
    )



# ---------------------------------------------------------------------------
# Probe 9: the residual threshold must come from an ORDERED walk, not a sum.
#
# This is the discriminator the ordered-walk rewrite shipped without. Nothing
# else in this file distinguishes the two threshold models: probe 7's oracle
# deliberately sums damage only ("a fixture carrying Leftovers or a threshold
# berry would net the heal against the tick and mis-measure it"), so every one of
# its fixtures avoids the members the rewrite adds. Reverting
# `residual_phase_final_hp` to the old sum passed the entire suite. Found in
# review of PR #1066.
#
# Fixture: Fearow 123/244, BURNED, holding LEFTOVERS. The phase heals at 10.4
# BEFORE it burns at 10.6, so:
#   ground truth  roll 108 -> 15 left -> +15 Leftovers -> -30 burn = 0   DIES
#                 roll 107 -> 16 left -> +15 Leftovers -> -30 burn = 1   LIVES
#   ordered walk  h* = 16, threshold = 123 - 16 + 1 = 108   CORRECT
#   damage sum    123 - (30 - 0) = 93; min roll 103 >= 93 so it does NOT split,
#                 and collapses to a single non-crit arm at 112
# So the sum model both mis-places the threshold by 15 and loses the split.
# ---------------------------------------------------------------------------
def probe_residual_ordered_walk() -> None:
    def build(weather, turns_remaining, item, status, hp):
        attacker = pe.Pokemon(
            id="gligar", level=81,
            types=("ground", "flying"), base_types=("ground", "flying"),
            hp=205, maxhp=205, ability="none", item="none",
            attack=170, defense=160, special_attack=120,
            special_defense=130, speed=250,
            moves=[pe.Move(id="rockslide", pp=16)],
        )
        defender = pe.Pokemon(
            id="fearow", level=81,
            types=("normal", "flying"), base_types=("normal", "flying"),
            hp=hp, maxhp=244, ability="none", item=item,
            attack=170, defense=145, special_attack=110,
            special_defense=125, speed=100, status=status,
            moves=[pe.Move(id="splash", pp=16)],
        )
        kw = {}
        if turns_remaining is not None:
            kw["weather_turns_remaining"] = turns_remaining
        return pe.State(
            side_one=pe.Side(active_index="0", pokemon=[attacker] + [_dummy()] * 5),
            side_two=pe.Side(active_index="0", pokemon=[defender] + [_dummy()] * 5),
            weather=weather, terrain="none", trick_room=False, **kw
        )

    def arms(**kw):
        branches = pe.generate_instructions(build(**kw), "rockslide", "splash")
        out = []
        for b in branches:
            dealt = [
                int(str(i).split(": ")[1])
                for i in b.instruction_list
                if str(i).startswith("Damage SideTwo")
            ]
            out.append((round(b.percentage, 4), dealt[0] if dealt else None))
        return out

    def split_values(arm_list, hp):
        """Distinct non-crit hit damages. Rock Slide flinches 30% of the time, so
        every arm appears twice and a raw branch COUNT is not a usable signal --
        an earlier version of this probe asserted 4 and saw 6. Drop the 10% miss
        arm and the crit arm (which deals exactly `hp`), then count distinct
        damages: 2 means the fan was partitioned, 1 means it was collapsed."""
        return sorted(
            {
                d
                for pct, d in arm_list
                if d is not None and d != hp and abs(pct - 10.0) > 1e-9
            }
        )

    # --- the motivating case: heal at 10.4 precedes the burn at 10.6 ----------
    burn_leftovers = arms(weather="none", turns_remaining=None,
                          item="leftovers", status="burn", hp=123)
    damages = sorted(d for _, d in burn_leftovers if d is not None)
    _report(
        "ordered-walk-heal-before-damage",
        108 in damages,
        f"Leftovers (+15 at 10.4) heals BEFORE the burn (-30 at 10.6), so the "
        f"residual-lethal threshold is 108, not the damage-sum's 93. Expected an "
        f"arm at exactly 108; got arms {burn_leftovers}. A single non-crit arm at "
        f"112 means the threshold came from a sum.",
    )
    _report(
        "ordered-walk-splits-the-fan",
        split_values(burn_leftovers, 123) == [105, 108],
        f"the fan straddles 108 (min roll 103), so it must partition into a "
        f"surviving representative and the threshold: expected non-crit damages "
        f"[105, 108], got {split_values(burn_leftovers, 123)} -- {burn_leftovers}",
    )

    # --- the weather decrement runs BEFORE the chip ---------------------------
    # turns_remaining == 1 expires during upkeep and never chips, so nothing is
    # pending and the fan must NOT split. 5 and -1 (permanent) both chip.
    expiring = arms(weather="sand", turns_remaining=1, item="none", status="none", hp=123)
    lasting = arms(weather="sand", turns_remaining=5, item="none", status="none", hp=123)
    permanent = arms(weather="sand", turns_remaining=-1, item="none", status="none", hp=123)
    _report(
        "ordered-walk-expiring-weather-does-not-chip",
        len(split_values(expiring, 123)) == 1,
        f"sand with turns_remaining=1 is decremented to 0 and ENDS before the "
        f"chip, so nothing is pending and the fan must not split: expected ONE "
        f"non-crit damage, got {split_values(expiring, 123)} -- {expiring}. "
        f"weather_is_active ignores "
        f"turns_remaining, so a mirror consulting it alone over-counts here.",
    )
    _report(
        "ordered-walk-lasting-weather-chips",
        len(split_values(lasting, 123)) == 2 and len(split_values(permanent, 123)) == 2,
        f"sand that survives upkeep must chip and split -- turns_remaining=5 gave "
        f"{split_values(lasting, 123)}, permanent gave "
        f"{split_values(permanent, 123)}, expiring gave "
        f"{split_values(expiring, 123)}. If these match the expiring case the "
        f"guard is inert.",
    )

    # --- the bail set gives up a split for every wrong entry ------------------
    # Salac only boosts a stat; Chesto's arm is gated on SLEEP (no 10.6 tick);
    # Shell Bell is not in item_end_of_turn at all. All three must still split.
    for item in ("salacberry", "chestoberry", "shellbell"):
        held = arms(weather="none", turns_remaining=None, item=item,
                    status="burn", hp=140)
        _report(
            f"ordered-walk-inert-item-still-splits-{item}",
            len(split_values(held, 140)) == 2,
            f"{item} does not change HP at end of turn, so declining for it "
            f"silently gives up a split: expected two non-crit damages, got "
            f"{split_values(held, 140)} -- {held}",
        )



# ---------------------------------------------------------------------------
# Probe 10: Case A must partition THREE ways, not two.
#
# When the fan straddles the hit-KO threshold, the engine splits kill / non-kill
# and collapses the non-kill side to `average_non_kill_damage`. That sub-fan can
# ALSO straddle the residual threshold, and Case A used never to consult it -- so
# when the collapsed representative landed on the lethal side of the residual,
# NO branch survived the turn at all.
#
# Seed 19000052 step 36: Walrein 192/307 burned holding Leftovers, with
# max_damage_dealt == 192 == hp. Ordered threshold 173 (Leftovers +19 at 10.4,
# burn -38 at 10.6). Six of sixteen rolls survive the residual and Showdown rolled
# the lowest of them, 163, finishing on 10 hp -- but the representative was 177,
# above 173, so the engine asserted a burn KO on every non-crit roll.
#
# Fixture: Fearow 120/244 burned + Leftovers, non-crit fan [103, 122].
#   103 < 120 <= 122          -> Case A
#   burn -30, Leftovers +15   -> h* = 16, residual threshold 105
#   103 < 105 < 120           -> the surviving sub-fan straddles it too
# So three distinct non-crit damages must appear: 103 (survives both), 105 (the
# residual threshold) and 120 (the hit KO). Two means Case A collapsed.
# ---------------------------------------------------------------------------
def probe_case_a_three_way() -> None:
    def damages(hp, status, item):
        attacker = pe.Pokemon(
            id="gligar", level=81,
            types=("ground", "flying"), base_types=("ground", "flying"),
            hp=205, maxhp=205, ability="none", item="none",
            attack=170, defense=160, special_attack=120,
            special_defense=130, speed=250,
            moves=[pe.Move(id="rockslide", pp=16)],
        )
        defender = pe.Pokemon(
            id="fearow", level=81,
            types=("normal", "flying"), base_types=("normal", "flying"),
            hp=hp, maxhp=244, ability="none", item=item,
            attack=170, defense=145, special_attack=110,
            special_defense=125, speed=100, status=status,
            moves=[pe.Move(id="splash", pp=16)],
        )
        state = pe.State(
            side_one=pe.Side(active_index="0", pokemon=[attacker] + [_dummy()] * 5),
            side_two=pe.Side(active_index="0", pokemon=[defender] + [_dummy()] * 5),
            weather="none", terrain="none", trick_room=False,
        )
        # Identify the MOVE damage structurally: it is the damage that lands
        # before any heal. Filtering the miss arm by percentage does not work --
        # its mass is not exactly 10% here, and its first `Damage SideTwo` is the
        # burn tick (30), which then reads as a fourth "roll". Rock Slide also
        # flinches 30%, duplicating every arm, so a branch COUNT is unusable too.
        seen = set()
        for b in pe.generate_instructions(state, "rockslide", "splash"):
            for i in b.instruction_list:
                text = str(i)
                if text.startswith("Heal SideTwo"):
                    break  # residual phase reached without the move connecting
                if text.startswith("Damage SideTwo"):
                    seen.add(int(text.split(": ")[1]))
                    break
        return sorted(seen)

    three_way = damages(120, "burn", "leftovers")
    _report(
        "case-a-partitions-three-ways",
        three_way == [103, 105, 120],
        f"the surviving sub-fan straddles the residual threshold 105, so Case A "
        f"must emit survives-both (103), residual-lethal (105) and hit-lethal "
        f"(120): got {three_way}. Two values means the non-kill side was "
        f"collapsed without consulting the residual threshold.",
    )

    # Control: same fixture with nothing pending. Case A must still split kill /
    # non-kill, but there is no third arm -- so exactly two values. This is what
    # isolates the residual partition as the variable rather than Case A itself.
    two_way = damages(120, "none", "none")
    _report(
        "case-a-control-has-no-third-arm",
        len(two_way) == 2 and two_way[-1] == 120,
        f"with no residual pending, Case A must split only kill / non-kill: "
        f"expected two damages ending in 120, got {two_way}. If this has three "
        f"the fixture is not isolating the residual arm.",
    )


# ---------------------------------------------------------------------------
# The CRIT-STRADDLE residual sub-split (c133 §3, dev row 19000074/27).
#
# When the crit fan straddles the hit-KO threshold while the non-crit fan does
# not, Case B split the crit mass into kill / non-kill and never consulted the
# residual threshold. The residual sub-split therefore existed at Case A and at
# the crit fan that CANNOT kill, and was missing from exactly this site.
#
# Seed 19000074 step 27: the priced crit fan is
# [214,216,219,221,224,226,229,231,234,236,239,241,244,246,249,252] against 244 HP
# with a sandstorm threshold of 229. Showdown rolled 241 -- roll 96, a member of
# the engine's own fan -- while the engine emitted arms only at 244 (the HP) and
# 227 (the mean of the twelve non-KO rolls, not a fan member at all).
#
# Fixture: Fearow 230/244 in sand, Rock Slide, non-crit fan [103, 122], crit fan
# [207, 244].
#   122 < 230                 -> Case B (the non-crit fan cannot kill)
#   207 < 230 <= 244          -> the CRIT fan straddles the hit-KO threshold
#   sand -15                  -> h* = 16, residual threshold 215
#   207 < 215 < 230           -> the surviving crit sub-fan straddles it too
#   215 > 122                 -> the non-crit fan cannot reach it, so this
#                                fixture isolates the crit site
# So four move damages must appear: 112 (the collapsed non-crit representative),
# 210 (crit, survives both), 215 (crit, residual-lethal) and 230 (crit, hit KO).
# Three -- with 217 in place of 210/215 -- is the collapsed crit sub-fan.
# ---------------------------------------------------------------------------
def probe_crit_straddle_residual_split() -> None:
    def move_damages(weather):
        def build(move):
            attacker = pe.Pokemon(
                id="gligar", level=81,
                types=("ground", "flying"), base_types=("ground", "flying"),
                hp=205, maxhp=205, ability="none", item="none",
                attack=170, defense=160, special_attack=120,
                special_defense=130, speed=250,
                moves=[pe.Move(id=move, pp=16)],
            )
            defender = pe.Pokemon(
                id="fearow", level=81,
                types=("normal", "flying"), base_types=("normal", "flying"),
                hp=230, maxhp=244, ability="none", item="none",
                attack=170, defense=145, special_attack=110,
                special_defense=125, speed=100, status="none",
                moves=[pe.Move(id="splash", pp=16)],
            )
            return pe.State(
                side_one=pe.Side(active_index="0", pokemon=[attacker] + [_dummy()] * 5),
                side_two=pe.Side(active_index="0", pokemon=[defender] + [_dummy()] * 5),
                weather=weather, terrain="none", trick_room=False,
            )

        # The residual tick, measured from a turn where neither side attacks.
        # Needed to identify the MISS arm: with no Leftovers there is no heal to
        # break on, so the miss arm's first `Damage SideTwo` IS the sand tick and
        # would otherwise read as a fourth roll. Rock Slide also flinches 30 %,
        # so a branch count is unusable here too.
        quiet = pe.generate_instructions(build("splash"), "splash", "splash")[0]
        tick = [
            int(str(i).split(": ")[1])
            for i in quiet.instruction_list
            if str(i).startswith("Damage SideTwo")
        ]
        tick = tick[0] if tick else None

        seen = set()
        for b in pe.generate_instructions(build("rockslide"), "rockslide", "splash"):
            hits = [
                int(str(i).split(": ")[1])
                for i in b.instruction_list
                if str(i).startswith("Damage SideTwo")
            ]
            if not hits:
                continue
            if len(hits) == 1 and hits[0] == tick:
                continue  # the miss arm: the residual tick and nothing else
            seen.add(hits[0])
        return sorted(seen)

    partitioned = move_damages("sand")
    _report(
        "crit-straddle-partitions-three-ways",
        partitioned == [112, 210, 215, 230],
        f"the surviving crit sub-fan straddles the residual threshold 215, so the "
        f"crit-straddle site must emit survives-both (210), residual-lethal (215) "
        f"and hit-lethal (230) alongside the collapsed non-crit 112: got "
        f"{partitioned}. A 217 in place of 210 and 215 is the mean of the ten "
        f"non-KO crit rolls -- the sub-fan collapsed without consulting the "
        f"residual threshold.",
    )

    # Control: the same fixture with no weather. The crit-KILL split must still
    # fire (that is C27, not this patch), so exactly three damages and no residual
    # arm. This is what isolates the residual sub-split as the variable rather
    # than the crit straddle itself.
    control = move_damages("none")
    _report(
        "crit-straddle-control-has-no-residual-arm",
        control == [112, 217, 230],
        f"with no residual pending the crit straddle must split only kill / "
        f"non-kill: expected [112, 217, 230], got {control}. If this equals the "
        f"sand case the fixture is not measuring the residual sub-split at all.",
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
    probe_contact_flags()
    probe_residual_lethality_partition()
    probe_residual_partition_masses()
    probe_pending_read_sees_this_calls_mutations()
    probe_residual_ordered_walk()
    probe_case_a_three_way()
    probe_crit_straddle_residual_split()
    if FAILURES:
        print(f"\n{len(FAILURES)} probe(s) FAILED — the installed wheel does "
              "not behave like the 33-patch engine. Rebuild before measuring.")
        return 1
    print("\nall behavioral probes PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
