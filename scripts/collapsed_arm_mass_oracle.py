#!/usr/bin/env python
"""The enumeration oracle for the collapsed roll-partition machinery — C138.

WHY THIS EXISTS. The collapsed damage path replaces a sixteen-roll fan with a
handful of representative arms. Three hand-derived mass recipes in this family have
already been wrong (C134 §3 froze it for that reason), and every one of them was
**mass-conserving**: totals summed to 100 % while the wrong rolls sat in the wrong
arms, so neither ``test_masses_sum_to_one`` nor the transition differential could
see them. Review had to substitute for the instrument.

The enumerate-then-merge spike removed that excuse. It emits one arm per distinct
``floor(max * r / 100)`` for ``r`` in ``85..=100`` at mass 1/16 and resolves
lethality, secondaries and the ordered residual phase inside ``run_move`` rather
than in a mirror, so for any fixture it is an EXACT reference for what the
collapsed path is approximating. This module makes that reference a test input.

THE FUNCTIONAL IS OUTCOME MASS, DELIBERATELY. A correct collapsed path *cannot*
agree with enumerated truth arm-for-arm — that is what collapsing means. The
comparison has to be a coarsening, and the one the disjoint-band recipe is exact
on is the total probability mass landing on each ``(defender faints?, defender's
end status)`` cell. That is the functional the spike's own A8 demonstration uses
(enumerated 5.810547 % against independent truth 5.810547 %, delta 0, where the
collapsed path gives 5.312500 %). Do not "strengthen" this into an arm-for-arm
comparison; it can never pass.

THE ORACLE CANNOT BE TOGGLED IN-PROCESS. ``ENUMERATE_DAMAGE_ROLLS`` is a
``OnceLock`` initialised from ``std::env::var`` on first call, so one process is
one engine, permanently. The enumerated side is therefore produced by running THIS
FILE as a script against a SEPARATE build in a SEPARATE venv, with
``POKEZERO_ENUMERATE_ROLLS=1``, and committed as
``tests/data/collapsed_arm_mass_oracle.json``. Pinning it is also what stops a
wrong oracle from silently blessing a wrong recipe — c137 §4's open item.

THREE-WAY, NOT TWO-WAY. ``tests/test_collapsed_arm_mass_oracle.py`` compares the
shipping engine's functional, the pinned enumerated functional, and
:func:`reconstruct_outcome_masses` — a pure-Python enumeration that shares no
partition arithmetic with either. It takes the residual phase's verdict from a
QUIET TURN at the exact post-move HP rather than from a scalar tick, so it carries
none of the "read at pre-move HP" imprecision the older mass gate documents.

Regenerate with::

    POKEZERO_ENUMERATE_ROLLS=1 <oracle-venv>/bin/python \\
        scripts/collapsed_arm_mass_oracle.py --write tests/data/collapsed_arm_mass_oracle.json

run from a checkout whose patch stack ends with
``poke-engine-gen3-enumerate-damage-rolls.patch``. The script refuses to write
unless the flag is actually in effect, measured behaviourally.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Mapping

import poke_engine as pe


BASE_CRIT_CHANCE = 1.0 / 16.0


@dataclasses.dataclass(frozen=True)
class CollapseFixture:
    """One boundary, described by the values that move the partition.

    Every field is a battle-legal value; ``hp <= maxhp`` is asserted when the state
    is built, because the constructor accepts an over-full defender silently and
    that is how a fixture once came to assert ``0.0 == 0.0`` while reading PASS.
    """

    label: str
    move: str
    accuracy: float
    hp: int
    maxhp: int
    status: str
    item: str
    weather: str
    special_attack: int
    special_defense: int
    attack: int
    defense: int
    # WHICH PARTITION SITE this fixture reaches. There are four, and the test
    # asserts every one of them appears -- coverage is machine-checked rather than
    # left to be re-derived by mutation, which is how the first version's gap was
    # found. See PARTITION_SITES.
    site: str
    # What the fixture is FOR, within that site. Asserted structurally by the test
    # so a fixture that stops exercising its shape fails by name.
    exercises: str


#: The four places `generate_instructions_from_move` partitions a damage fan.
#: Named here so the coverage assertion can be written against a list rather than
#: against whatever the fixtures happen to reach.
#:
#:   case-a               `ko_max_damage >= hp && ko_min_damage < hp` -- the fan
#:                        straddles the hit-KO threshold. A KO arm already exists,
#:                        so `hp` is the band CEILING.
#:   case-b-noncrit       Case B's non-crit fan, which cannot kill on the hit.
#:                        Unbounded ceiling.
#:   case-b-crit-straddle Case B's crit fan when it straddles the hit-KO
#:                        threshold. Ceiling `hp`, like case-a.
#:   case-b-crit-nokill   Case B's crit fan when it cannot kill either. Unbounded.
#:
#: An off-by-one at any one of these must turn some fixture RED; §"coverage" in
#: the test asserts the mapping is total, and the report records the mutation runs
#: that prove each fixture actually reaches its site.
PARTITION_SITES = (
    "case-a",
    "case-b-noncrit",
    "case-b-crit-straddle",
    "case-b-crit-nokill",
)

#: The attacker is Ground/Flying so it is immune to the sandstorm the weather
#: fixtures need. The defender is Normal/Flying: not Fire (so burn lands), not
#: Poison or Steel (so Toxic lands), and not Ground/Rock/Steel (so sand chips it).
FIXTURES: tuple[CollapseFixture, ...] = (
    # (A) The crit-straddle site. Rock Slide's non-crit max is 122 against 230 HP,
    # so Case B is taken; the crit fan
    # [207,209,212,214,217,219,222,224,226,229,231,234,236,239,241,244] straddles
    # 230, and the sandstorm threshold 215 sits INSIDE the surviving crit sub-fan.
    # The non-crit fan cannot reach 215, so this fixture isolates the crit site.
    CollapseFixture(
        label="crit-straddle-sand",
        move="rockslide", accuracy=0.9,
        hp=230, maxhp=244, status="none", item="none", weather="sand",
        special_attack=120, special_defense=125, attack=170, defense=145,
        site="case-b-crit-straddle", exercises="crit-straddle-residual",
    ),
    # (B) A8 in its pure form: unstatused defender holding Leftovers, no weather,
    # so the PRE-MOVE mirror declines and the split never fires today. Sacred
    # Fire's own 50 % burn makes 105 a live threshold inside the non-crit fan
    # [93..110]. The crit fan's floor (187) is above the defender's HP, so every
    # crit roll kills on the hit and the crit side is inert here.
    CollapseFixture(
        label="a8-burn-secondary",
        move="sacredfire", accuracy=0.95,
        hp=120, maxhp=244, status="none", item="leftovers", weather="none",
        special_attack=200, special_defense=125, attack=170, defense=145,
        site="case-b-noncrit", exercises="status-aware-threshold",
    ),
    # (B) The NESTED case, which is the only one that exercises the disjoint-band
    # rule as a rule. Sandstorm gives a pre-move threshold of 237 and the burn
    # secondary a lower one of 206, both strictly inside the fan
    # [204,206,208,211,213,216,218,220,223,225,228,230,232,235,237,240]: bands of
    # 13 and 2 with a survive arm of 1, totalling exactly 16/16. Pricing each arm
    # at #{rolls >= t_i} instead gives 15/16 + 2/16 + 1/16 = 18/16, which
    # `update_percentage(1 - branch_chance - residual_kill_chance)` has no room
    # for. Neither threshold is the fan's top rung, deliberately: the f32
    # comparator drops the top rung for 173 max values (C116 M5, still open), and
    # a fixture sitting on that rung would measure THAT defect instead of this
    # one.
    CollapseFixture(
        label="nested-thresholds",
        move="sacredfire", accuracy=0.95,
        hp=252, maxhp=255, status="none", item="none", weather="sand",
        special_attack=200, special_defense=57, attack=170, defense=145,
        site="case-b-noncrit", exercises="disjoint-bands-unbounded-ceiling",
    ),
    # (B) The control the MINIMUM-over-statuses recipe fails. The pre-move
    # threshold 104 is inside the fan; the burn threshold 88 is below the fan's
    # floor of 93. The union skips 88 and keeps the 104 arm the engine emits
    # today; `min(104, 88)` = 88 makes every `min_roll < threshold` guard false,
    # the partition stops firing entirely, and the 104 arm is destroyed. This
    # fixture must be GREEN before and after — it is the regression c133 §4
    # measured, not a target.
    CollapseFixture(
        label="min-would-destroy-an-arm",
        move="sacredfire", accuracy=0.95,
        hp=112, maxhp=128, status="none", item="none", weather="sand",
        special_attack=200, special_defense=125, attack=170, defense=145,
        site="case-b-noncrit", exercises="union-not-minimum",
    ),
    # Negative control: nothing pending, no reachable residual, so the fan stays
    # collapsed and only a crit kills. A matrix of straddles alone cannot tell a
    # correct partition from one that fires everywhere.
    CollapseFixture(
        label="collapsed-fan-control",
        move="rockslide", accuracy=0.9,
        hp=160, maxhp=244, status="none", item="none", weather="none",
        special_attack=120, special_defense=125, attack=170, defense=145,
        site="case-b-noncrit", exercises="negative-control",
    ),
    # (A)+(B) CASE A, with a THREE-LEVEL nest: the KO threshold, a pre-move
    # sandstorm threshold, and a lower Toxic threshold from the move's own
    # secondary. This is the only fixture that reaches the case-a site, and the
    # only one where the KO threshold acts as the band CEILING -- the correction
    # this branch makes to c135 section 5's recipe. `nested-thresholds` exercises
    # two bands with an UNBOUNDED ceiling; this exercises two bands under a KO
    # arm, which is different arithmetic.
    #
    # Poison Fang (30 % badly poison) at 223 max into 222/240 HP in sand:
    #   fan floor 189 < 222 <= 223  -> case-a (the fan straddles the hit KO)
    #   sand -15                    -> pre-move threshold 207
    #   sand -15 then toxic -15     -> status-aware threshold 192
    #   189 < 192 < 207 < 222
    # Bands: KO #{>= 222} = 1, [207, 222) = 7, [192, 207) = 6, survive = 2. The
    # four arms land on four DISTINCT outcomes -- dead on the hit, dead to sand,
    # dead only if the Toxic lands, and alive either way -- so the outcome-mass
    # functional sees all three boundaries at once. Pricing the top band at
    # #{>= 207} = 8 rather than 7 would steal the KO roll, which is exactly the
    # double-count c133 section 4 warns about.
    CollapseFixture(
        label="case-a-nested-ko-ceiling",
        move="poisonfang", accuracy=1.0,
        hp=222, maxhp=240, status="none", item="none", weather="sand",
        special_attack=120, special_defense=125, attack=390, defense=60,
        site="case-a", exercises="ko-threshold-is-the-band-ceiling",
    ),
    # (B) The CRIT FAN THAT CANNOT KILL -- the fourth site, which no other fixture
    # here reaches. Rock Slide into 250/255 HP in sand: the crit fan [207 .. 244]
    # tops out BELOW the defender's HP, so the crit-straddle arm is not taken and
    # the sibling `else` runs instead.
    #   122 < 250                   -> Case B
    #   crit max 244 < 250          -> the crit fan cannot kill on the hit either
    #   sand -15                    -> threshold 235, inside the crit fan
    #   235 > 122                   -> the NON-crit fan cannot reach it, so this
    #                                  fixture isolates the crit site
    # Four of the sixteen crit rolls are residual-lethal and nothing else on the
    # boundary faints, so the entire faint mass is this one arm: 0.9 * 1/16 * 4/16
    # = 1.40625 %. An arm priced one below the threshold leaves the defender on
    # 1 HP and the whole faint mass disappears.
    CollapseFixture(
        label="crit-fan-cannot-kill-sand",
        move="rockslide", accuracy=0.9,
        hp=250, maxhp=255, status="none", item="none", weather="sand",
        special_attack=120, special_defense=125, attack=170, defense=145,
        site="case-b-crit-nokill", exercises="crit-fan-residual-unbounded-ceiling",
    ),
)


def _dummy() -> pe.Pokemon:
    return pe.Pokemon(id="pikachu", level=1, hp=0)


def build_state(fixture: CollapseFixture, *, hp: int | None = None,
                status: str | None = None, move: str | None = None) -> pe.State:
    """The fixture's state, optionally with the defender's HP or status overridden.

    The overrides exist for :func:`reconstruct_outcome_masses`, which probes the
    residual phase at a specific post-move HP.
    """

    defender_hp = fixture.hp if hp is None else hp
    defender_status = fixture.status if status is None else status
    attacker_move = fixture.move if move is None else move
    if not 0 < defender_hp <= fixture.maxhp:
        raise ValueError(
            f"{fixture.label}: hp {defender_hp} outside (0, {fixture.maxhp}]; an "
            "over-full defender is an unreachable battle state that shifts every "
            "threshold while the constructor accepts it silently"
        )
    attacker = pe.Pokemon(
        id="gligar", level=81,
        types=("ground", "flying"), base_types=("ground", "flying"),
        hp=205, maxhp=205, ability="none", item="none",
        attack=fixture.attack, defense=160, special_attack=fixture.special_attack,
        special_defense=130, speed=250,
        moves=[pe.Move(id=attacker_move, pp=16), pe.Move(id="splash", pp=16)],
    )
    defender = pe.Pokemon(
        id="fearow", level=81,
        types=("normal", "flying"), base_types=("normal", "flying"),
        hp=defender_hp, maxhp=fixture.maxhp, ability="none", item=fixture.item,
        attack=170, defense=fixture.defense, special_attack=110,
        special_defense=fixture.special_defense,
        speed=100, status=defender_status,
        moves=[pe.Move(id="splash", pp=16)],
    )
    return pe.State(
        side_one=pe.Side(active_index="0", pokemon=[attacker] + [_dummy()] * 5),
        side_two=pe.Side(active_index="0", pokemon=[defender] + [_dummy()] * 5),
        weather=fixture.weather, terrain="none", trick_room=False,
    )


def _net_hp_lost_by_defender(branch) -> int:
    """NET HP the defender loses across a branch: damage minus heals.

    Damage-only was the older mass gate's most serious defect — a residual
    expressed as a heal was dropped identically on both sides, so the two agreed on
    a fictitious threshold.
    """

    total = 0
    for instruction in branch.instruction_list:
        text = str(instruction)
        if text.startswith("Damage SideTwo"):
            total += int(text.split(": ")[1])
        elif text.startswith("Heal SideTwo"):
            total -= int(text.split(": ")[1])
    return total


def _defender_end_status(branch, starting_status: str) -> str:
    status = starting_status
    for instruction in branch.instruction_list:
        text = str(instruction)
        if text.startswith("ChangeStatus SideTwo"):
            status = text.split("-> ")[1].strip().lower()
    return status


def outcome_masses(fixture: CollapseFixture, state: pe.State | None = None) -> dict[str, float]:
    """The functional: mass per ``faint|status`` cell, in percent.

    This is the ONLY comparison this module makes between two engines. See the
    module docstring for why it is a coarsening rather than an arm comparison.
    """

    if state is None:
        state = build_state(fixture)
    cells: dict[str, float] = {}
    for branch in pe.generate_instructions(state, fixture.move, "splash"):
        fainted = _net_hp_lost_by_defender(branch) >= fixture.hp
        status = _defender_end_status(branch, fixture.status)
        key = f"{'faint' if fainted else 'alive'}|{status}"
        cells[key] = cells.get(key, 0.0) + branch.percentage
    return cells


def _residual_kills(fixture: CollapseFixture, post_move_hp: int, status: str) -> bool:
    """Does the end-of-turn phase finish a defender sitting on ``post_move_hp``?

    Measured, not modelled: a quiet turn where NEITHER side attacks reports the
    phase's own verdict at exactly the HP in question. Taking it at the post-move
    HP rather than as a scalar read at pre-move HP is what removes the Leftovers
    ``min(maxhp/16, maxhp - hp)`` imprecision the older gate has to document.
    """

    if post_move_hp <= 0:
        return True
    quiet = build_state(fixture, hp=post_move_hp, status=status, move="splash")
    branches = pe.generate_instructions(quiet, "splash", "splash")
    return any(_net_hp_lost_by_defender(b) >= post_move_hp for b in branches)


def _secondary_status(fixture: CollapseFixture) -> tuple[str, float]:
    """The status secondary the fixture's move carries, as ``(status, chance)``.

    Hardcoded rather than read from the engine, which is the point: reading it back
    out of ``choices.rs`` would destroy the independence this reconstruction has
    left. Sacred Fire burns 50 % of the time in gen 3; Rock Slide's only secondary
    is a flinch, which changes no HP.
    """

    return {
        "sacredfire": ("burn", 0.5),
        "poisonfang": ("toxic", 0.3),
    }.get(fixture.move, ("none", 0.0))


def reconstruct_outcome_masses(fixture: CollapseFixture) -> dict[str, float]:
    """The same functional, rebuilt from integer roll enumeration in Python.

    Shares ``calculate_damage``'s VALUE and the residual phase's VERDICT with the
    engine — neither is reproducible here and pretending otherwise is how a
    previous gate came to claim independence it did not have. What is independent
    is everything the partition does: the roll enumeration, the per-roll
    classification, the secondary composition and the mass formula.
    """

    state = build_state(fixture)
    max_regular = pe.calculate_damage(state, fixture.move, "splash", False)[0][0]
    max_crit = pe.calculate_damage(state, fixture.move, "splash", True)[0][1]
    secondary_status, secondary_chance = _secondary_status(fixture)

    cells: dict[str, float] = {}

    def add(key: str, mass: float) -> None:
        if mass:
            cells[key] = cells.get(key, 0.0) + mass * 100.0

    # The miss arm: no damage, the phase still runs.
    miss_faints = _residual_kills(fixture, fixture.hp, fixture.status)
    add(f"{'faint' if miss_faints else 'alive'}|{fixture.status}", 1.0 - fixture.accuracy)

    for maximum, fan_mass in (
        (max_regular, fixture.accuracy * (1.0 - BASE_CRIT_CHANCE) / 16.0),
        (max_crit, fixture.accuracy * BASE_CRIT_CHANCE / 16.0),
    ):
        for roll in range(85, 101):
            damage = maximum * roll // 100
            if damage >= fixture.hp:
                # Dead on the hit; `immune_to_status` refuses a status at hp <= 0,
                # so the secondary cannot land on this roll.
                add(f"faint|{fixture.status}", fan_mass)
                continue
            remaining = fixture.hp - damage
            for status, chance in (
                (secondary_status, secondary_chance),
                (fixture.status, 1.0 - secondary_chance),
            ):
                if not chance:
                    continue
                faints = _residual_kills(fixture, remaining, status)
                add(f"{'faint' if faints else 'alive'}|{status}", fan_mass * chance)
    return cells


def collect(fixtures=FIXTURES) -> dict[str, dict[str, float]]:
    return {fixture.label: outcome_masses(fixture) for fixture in fixtures}


def enumeration_is_active() -> bool:
    """Measure the flag behaviourally; never trust the environment variable alone.

    Under enumeration the engine emits one arm per DISTINCT roll, so the
    negative-control fixture — which collapses to a single non-crit arm and a
    single crit arm — comes back with many more branches. Reading
    ``POKEZERO_ENUMERATE_ROLLS`` instead would let a build without the patch write
    a "enumerated" artifact that is really the collapsed path compared against
    itself.
    """

    control = next(f for f in FIXTURES if f.exercises == "negative-control")
    branches = pe.generate_instructions(build_state(control), control.move, "splash")
    return len(branches) > 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    if not enumeration_is_active():
        print(
            "ERROR: this build is not enumerating. Apply "
            "third_party/poke-engine-gen3-enumerate-damage-rolls.patch, rebuild "
            "into a separate venv, and set POKEZERO_ENUMERATE_ROLLS=1.",
            file=sys.stderr,
        )
        return 2

    payload: Mapping[str, object] = {
        "_README": (
            "Enumerated ground truth for tests/test_collapsed_arm_mass_oracle.py. "
            "One arm per distinct floor(max*r/100), r in 85..=100, mass 1/16, with "
            "lethality resolved inside run_move. Regenerate with "
            "scripts/collapsed_arm_mass_oracle.py --write from a build carrying "
            "poke-engine-gen3-enumerate-damage-rolls.patch. NEVER regenerate it "
            "from the shipping build to make a test pass: it is the reference."
        ),
        "functional": "probability mass in percent per (defender faints?, defender end status)",
        "fixtures": {f.label: dataclasses.asdict(f) for f in FIXTURES},
        "outcome_masses": collect(),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write is None:
        print(text, end="")
    else:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
