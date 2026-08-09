"""Standing mass gate — C116 Phase 1 item 4.

WHY THIS EXISTS. The engine transition differential compares roll-scaled damage
*components*; it never compares branch *probability masses*. So an entire class of
defect is invisible to it: anything that moves mass between arms while leaving each
arm's components intact. On PR #1062 the non-crit residual split called
``update_percentage`` in place, silently scaling every crit arm cloned from that value
afterwards, and the fix measured NEUTRAL on a 200-game sweep. Adversarial review has
had to substitute for this instrument three separate times (that mass leak, a
cross-gen ``cfg!`` edit, and a rewrite of the whole threshold model that no test
distinguished). Review is not an instrument. This is the instrument.

A CLAIM WITHDRAWN, because it was this file's original stated reason to exist and it
was false. The docstring used to say the #1062 leak left totals summing to 100% so no
conservation check would fire. Review measured totals of 95.78–97.89 on the affected
fixtures: ANY in-place early reduction loses ``crit_rate * n/16``, so that leak is
caught by the free ``test_masses_sum_to_one`` and never motivated this gate. What
motivates it is MASS-CONSERVING error — a threshold off by one, or a residual mirror
that misplaces the threshold — which holds the total at 100% while putting the wrong
rolls in the wrong arms. This PR's red run is such a mutant, deliberately.

WHAT IT ASSERTS. For each fixture, the engine's total "defender dies this turn"
probability mass must equal a reconstruction that is INDEPENDENT IN THE PARTS THAT
MATTER. Be precise, because "shares no arithmetic" was claimed here and was false: the
damage formula comes from ``calculate_damage`` and the residual magnitude from the
phase itself, so those ARE shared. What is independent is the roll enumeration, the
per-roll classification, and the mass formula — everything the partition logic does.
This gate cannot catch a wrong damage formula; it catches wrong ARM ASSIGNMENT and
wrong MASSES.

  1. enumerate the sixteen gen3 rolls as ``floor(max * r / 100)`` for r in 85..=100,
     taking ``max`` from ``calculate_damage`` (a value, not a code path);
  2. read the residual NET (damage minus heals) from a turn where NEITHER side
     attacks, so the phase reports its own magnitude rather than us predicting it.
     Two known imprecisions, recorded rather than fixed: it is read at PRE-move HP
     while a Leftovers heal is ``min(maxhp/16, maxhp - hp)`` at POST-move HP, so it
     understates the healing available to a damaged defender — the same trap the
     shipped mirror's own comment flags — and it is a scalar, so it cannot represent a
     non-monotone threshold heal. Harmless while ``maxhp - hp >= 84`` in every fixture
     here; a real limit if fixtures move closer to full HP;
  3. count the rolls that die, non-crit and crit separately;
  4. mass = accuracy * ((1 - crit_rate) * n_regular/16 + crit_rate * n_crit/16).

Nothing here calls ``compare_health_with_damage_multiples``, the residual mirror, or
the partition logic. It also asserts every fixture's masses sum to 100%, which is
weaker but free — and which, per the withdrawal above, is what actually catches the
#1062 leak.

WHAT IT DOES NOT COVER, measured by mutation rather than guessed. Corrupting C27's
crit-kill split and #1062's crit-fan residual split BOTH to ``crit_rate*0.5`` left an
earlier version of this matrix entirely green: neither path was reached, because
``min_crit=207`` and ``max_crit=244`` while every fixture's hp sat outside
``(207, 244]``. ``crit-kill-straddle`` closes the first — measured by review of #1074
(mutation: ``crit_kill_chance`` and ``crit_residual_kill`` both set to
``crit_rate*0.5``), re-derived at 7b70d8a7, that fixture goes red at 2.8125% against a
reconstruction of 2.1094%. Attributed rather than restated, per the M2 rule: I did not
take that measurement.

Still uncovered BY ``CASES``: the crit-FAN residual split (needs ``max_crit < hp``, a
second attacker profile rather than another hp value), ``fixed_damage``, multi-hit
moves, the Wish / Rain Dish / Leech Seed / partial-trap mirror steps, and the bail set.
The bail set is unreachable BY THIS DESIGN — a scalar quiet-turn tick cannot represent
Sitrus's non-monotone threshold heal — so covering it needs a different reconstruction,
not another fixture. These are an obligation on the Phase 2 decision record: if the
partition stack is RETAINED for any consumer (plan outcomes (b) or (c)), they get
fixtures before that decision is recorded as closed.

THAT OBLIGATION IS DISCHARGED FOR THOSE FOUR FAMILIES, AND FOR NOTHING ELSE. ``CASES``
is byte-unchanged; the discharge is ``OWED_CASES`` below, which has its own comment
block, its own reconstruction and its own tests. Evidence, including the mutation run
that shows each new fixture reddens on its own and the pool-reachability measurements,
is ``reports/c152_mass_gate_owed_fixtures.md``. Two things are explicitly NOT
discharged and are carried forward rather than quietly dropped: **the bail set**, which
is the one entry above this reconstruction cannot express at all, and
**``mirror-step-rain-dish``**, which is discharged as pool-UNREACHABLE rather than by a
fixture -- see ``POOL_UNREACHABLE`` for the measurement. Shipping a Rain Dish fixture
would read as coverage of a phase step no gen3 randbats game can execute.

CI-GATING IS NOT DELIVERED. Nothing runs ``tests/`` wholesale — two workflows run six
named modules between them and neither builds ``poke_engine``. Plan item 4 asks for
CI-gating; this file is the standing half, and the wiring is a following PR. When it is
wired, the module-level ``import poke_engine`` must stay HARD: a gate that skips when
the wheel is missing is how the previous era's fixtures read PASS while asserting
nothing.

FIXTURE DESIGN, learned from six near-misses. A fixture that does not straddle a
threshold asserts nothing and reads PASS. ``test_matrix_is_not_vacuous`` asserts the
matrix contains a genuine split and a collapsed fan, and that ``case-a-three-way``
partitions. A branch COUNT is never used as a signal: Rock Slide flinches 30%, so every
arm appears twice and a count moves with an unrelated secondary.

WHICH ROLL PATH THIS GATE MEASURES, since C116 Phase 2 there are two. The engine now
ships enumerate-then-merge behind the runtime flag ``POKEZERO_ENUMERATE_ROLLS``, OFF by
default and consumed only as a reference ORACLE by tests that ask for it explicitly
(``tests/test_roll_enumeration_scope.py`` holds that line at runtime). This gate's
subject is the COLLAPSED partition cascade — the path search runs, the path the
transition differential runs, and the only path whose arm assignment a threshold can get
wrong, because enumeration consults no threshold at all.

The three mass assertions are configuration-agnostic and stay in-process: a KO mass, a
mass total and a named-constant pin are true of both paths, and measuring whichever path
the caller built is a feature. ``test_matrix_is_not_vacuous`` is NOT: its negative control
asserts that some fixture leaves a fan COLLAPSED, which is unsatisfiable by construction
under enumeration, where every fan is partitioned into its sixteen rolls. Under ambient
``POKEZERO_ENUMERATE_ROLLS=1`` the old version therefore went red for a reason that has
nothing to do with the defect class it guards.

The fix is not to relax it. The flag is latched per process by a ``OnceLock``, so this
gate measures its shapes in a CHILD process with ``POKEZERO_ENUMERATE_ROLLS=0`` and keeps
every assertion at full strength. Verified by running the control against a child with
the flag ON and confirming it fails.

What forcing ``"0"`` does and does not catch, stated precisely because an earlier version
of this paragraph overclaimed. It catches the flag being IGNORED — an engine that
enumerates whatever the environment says makes this child come back enumerated and the
collapsed control go red. It does NOT catch the patch's DEFAULT flipping, because the
value is set explicitly here and a default only applies when the variable is unset. That
property is asserted where it belongs, against an unset environment, by
``tests/test_roll_enumeration_scope.py::test_default_build_collapses_the_fan``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import poke_engine as pe

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _subproc_env import subproc_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENUMERATE_ROLLS_ENV = "POKEZERO_ENUMERATE_ROLLS"

ACCURACY = 0.9          # Rock Slide
CRIT_RATE = 1.0 / 16.0  # BASE_CRIT_CHANCE
MAXHP = 244


def _dummy() -> pe.Pokemon:
    return pe.Pokemon(id="pikachu", level=1, hp=0)


def _state(hp, status, item, weather, toxic_count, attacker_move, defender_speed=100):
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
        hp=hp, maxhp=MAXHP, ability="none", item=item,
        attack=170, defense=145, special_attack=110,
        special_defense=125, speed=defender_speed, status=status,
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


def _net_hp_lost_by_defender(branch) -> int:
    """NET HP the defender loses across the branch: damage minus heals.

    This was damage-only, and that was the gate's most serious defect. The same
    helper feeds both the reconstruction's tick AND the engine's KO set, so a
    residual expressed as a heal was dropped on both sides identically and the two
    agreed on a fictitious threshold. Measured consequence: a residual mirror that
    loses the 10.4 Leftovers heal -- the exact "damage-only SUM puts the threshold
    too low" error the shipped patch comment warns about -- made the engine assert a
    burn KO on 4 of 16 surviving rolls (true KO mass 68.91%, engine 90.00%, a
    21.1-point error) and this file stayed GREEN on all nine fixtures. The gate was
    blind to a live instance of its own target class. Found by review of #1074.
    """
    total = 0
    for i in branch.instruction_list:
        text = str(i)
        if text.startswith("Damage SideTwo"):
            total += int(text.split(": ")[1])
        elif text.startswith("Heal SideTwo"):
            total -= int(text.split(": ")[1])
    return total


def measure_shapes() -> dict[str, list[int]]:
    """{fixture -> sorted distinct move-damage values in its non-miss arms}.

    Damage VALUES, never branch counts: Rock Slide flinches 30%, so every arm
    appears twice and a count moves with an unrelated secondary.
    """
    shapes = {}
    for label, hp, status, item, weather, tc in BranchMassReconstruction.CASES:
        quiet = _state(hp, status, item, weather, tc, "splash")
        tick = _net_hp_lost_by_defender(
            pe.generate_instructions(quiet, "splash", "splash")[0]
        )
        state = _state(hp, status, item, weather, tc, "rockslide")
        values = set()
        for b in pe.generate_instructions(state, "rockslide", "splash"):
            # A MISS branch loses exactly the residual tick and nothing else;
            # any hit adds move damage on top, so comparing the NET against the
            # tick identifies it exactly.
            #
            # My first attempt broke on a marker instruction instead, and review
            # MEASURED that it did not work: for every fixture without Leftovers
            # the miss branch's FIRST instruction IS the bare residual
            # `Damage SideTwo`, so there was nothing to break on.
            # saturated-toxic-count-1 still read [30, 112], still misreporting a
            # collapsed fan as partitioned, while the comment claimed otherwise.
            # It now reads [112] and is the negative control it was designed to be.
            if _net_hp_lost_by_defender(b) == tick:
                continue
            for i in b.instruction_list:
                text = str(i)
                if text.startswith("Damage SideTwo"):
                    values.add(int(text.split(": ")[1]))
                    break
        shapes[label] = sorted(v for v in values if v != hp)
    return shapes


# ---------------------------------------------------------------------------
# C119 obligation 1, re-fired by ``reports/c137_phase2_enumerate_decision.md`` §4
# because Phase 2 RETAINED the partition stack: the four families this gate's
# "still uncovered" paragraph names get fixtures before that decision is recorded
# as closed. This block is that discharge.
#
# WHY A SECOND MATRIX RATHER THAN MORE ``CASES``. Every family here needs a knob
# ``CASES`` does not have -- a per-case move (``fixed_damage``, multi-hit), a
# per-case ``maxhp`` (the crit fan cannot fail to kill while ``max_crit == maxhp``),
# a volatile status (Leech Seed, partial trap) or a pending Wish. Widening the
# six-tuple would have rewritten the negative control's shape probe and the two
# pins that name it, for fixtures that share none of their arithmetic. ``CASES``
# is therefore untouched and this matrix carries its own reconstruction.
#
# WHAT THE RECONSTRUCTION ADDS OVER ``CASES``'s. Two descriptors, both properties
# of the MOVE and neither read out of the partition code:
#   ``rolls``  False for a ``damageCallback`` move, which has no 85-100 fan and
#              cannot crit. ``calculate_damage`` returns a ONE-element list for
#              those and a two-element one otherwise; the size is asserted, so a
#              move that silently grows a crit entry fails by name rather than
#              being read as ``damages[1]`` by accident.
#   ``hits``   The comparator is handed a TOTAL for a multi-hit move, so the fan
#              this gate enumerates must be the total too. Multiplying here is what
#              makes the gate see the hit-count scaling at all.
# ---------------------------------------------------------------------------

#: The families ``reports/c137_phase2_enumerate_decision.md`` §4 and this module's
#: docstring both name. The fourth bullet is one sentence in both sources -- "the
#: Wish / Rain Dish / Leech Seed / partial-trap mirror steps" -- and is split into
#: one entry per PHASE STEP here, because they are four independent branches of
#: ``residual_phase_final_hp`` (orders 7, 10.3, 10.5 and 10.9) and a single fixture
#: reaching one of them would have satisfied a single-entry check while leaving the
#: other three unguarded. The bail set is deliberately ABSENT: the docstring's
#: uncovered list has it as a fifth entry and rules it out of fixture scope in the
#: same sentence, and c137 §4's own list of four omits it.
OWED_FAMILIES = (
    "crit-fan-residual-split",
    "fixed-damage",
    "multi-hit",
    "mirror-step-wish",
    "mirror-step-rain-dish",
    "mirror-step-leech-seed",
    "mirror-step-partial-trap",
)

#: The committed pool census the reachability verdicts below are read from.
#: Produced out of process by ``scripts/c152_pool_reachability_census.py`` against a
#: pokemon-showdown checkout, because CI builds none and this module's workflow step
#: forbids skips outright -- so a live re-derivation cannot run here.
POOL_CENSUS_PATH = ROOT / "tests" / "data" / "c152_pool_reachability_census.json"

#: Families discharged by a REACHABILITY measurement instead of a fixture, in the
#: convention of ``reports/c138_known_gaps_ledger.md`` §1.1: the verdict AND the
#: instrument that produced it. A fixture for an unreachable step reads as coverage
#: of behaviour no game can execute, which is the "inert pin" shape this repository
#: has found five times.
#:
#: ``counts`` is the load-bearing part and is FIGURES, not prose, deliberately. An
#: earlier version of this waiver was prose alone and c137's bullet described it as
#: "machine-checked" -- which was false as stated: the only assertion checked that
#: the family had *a* verdict, not that Rain Dish is absent. The figures below are
#: asserted against ``POOL_CENSUS_PATH`` by
#: ``test_the_rain_dish_waiver_is_backed_by_the_committed_census``.
POOL_UNREACHABLE = {
    "mirror-step-rain-dish": {
        "verdict": "UNREACHABLE",
        "instrument": (
            "the union of every set's `abilities` in "
            "`data/random-battles/gen3/sets.json`, per c138 §1.2"
        ),
        "counts": {
            "showdown_commit": "f76228a1354b5d0f307ca2d16101294ad3a2308b",
            "species": 220,
            "sets": 393,
            "distinct_abilities": 71,
            # The verdict itself: zero pool sets list either order-10.3 healer.
            "Rain Dish": 0,
            # The mirror's own comment names DRYSKIN as the other HP-changing
            # ability at 10.3, so a waiver measuring only Rain Dish would be
            # narrower than the step it waives.
            "Dry Skin": 0,
        },
        "note": (
            "Trace IS in the 71 and cannot manufacture it: Trace copies the "
            "OPPONENT's ability, and no pool member has Rain Dish to copy -- the "
            "cross-side check c138's R26 correction requires. The step is "
            "gen3-legal (`Dex.mod('gen3').abilities.get('raindish')` resolves, "
            "num 44) and live in the mirror, so it is unreachable in the POOL, "
            "not in gen3."
        ),
    },
}


@dataclasses.dataclass(frozen=True)
class OwedCase:
    """One owed family's fixture. Every field is a battle-legal value.

    ``covers`` is machine-checked against :data:`OWED_FAMILIES`, so a family cannot
    lose its only fixture silently, and ``exercises`` is asserted structurally by
    :meth:`BranchMassReconstruction.test_the_owed_matrix_is_not_vacuous` so a
    fixture that drifts off the shape it is named for fails by name.
    """

    label: str
    covers: str
    exercises: str
    move: str
    accuracy: float
    hits: int
    rolls: bool
    hp: int
    maxhp: int
    status: str = "none"
    item: str = "none"
    weather: str = "none"
    toxic_count: int | None = None
    volatiles: tuple[str, ...] = ()
    wish: tuple[int, int] = (0, 0)
    defender_types: tuple[str, str] = ("normal", "flying")


OWED_CASES: tuple[OwedCase, ...] = (
    # (1) THE CRIT-FAN RESIDUAL SPLIT. The docstring says this needs ``max_crit <
    # hp`` and "a second attacker profile rather than another hp value" -- that is
    # half right and the cheaper half is enough: raising the DEFENDER's maxhp above
    # the crit fan's top does it, because ``CASES``'s crit fan tops out at exactly
    # MAXHP=244 and no hp <= 244 can sit above it. At maxhp 255 the crit fan
    # [207..244] cannot kill on the hit against 250 HP, so the crit-STRADDLE arm is
    # not taken and its sibling `else` -- the fourth partition site, and the one no
    # fixture in ``CASES`` reaches -- runs instead. Sand ticks 15, so the threshold
    # is 235 and four of the sixteen crit rolls die to it; the non-crit fan tops at
    # 122 and cannot reach 235, which is what isolates the crit site. The entire
    # faint mass is that one arm, 0.9 * 1/16 * 4/16 = 1.40625 %.
    OwedCase(
        label="crit-fan-residual-sand",
        covers="crit-fan-residual-split",
        exercises="crit fan cannot kill on the hit; residual threshold inside it",
        move="rockslide", accuracy=0.9, hits=1, rolls=True,
        hp=250, maxhp=255, weather="sand",
    ),
    # (2) FIXED DAMAGE. Seismic Toss deals `level` (81) with no 85-100 roll and no
    # crit, so the whole fan is one point. hp and the burn tick are chosen so that
    # this is the ONLY reason the answer is 100 %: the threshold is 76, the fixed 81
    # clears it, and a roll-scaled Seismic Toss would fan [68..81] with seven rolls
    # below the threshold -- 50.0000 % instead of 100.0000 %. So the fixture is a
    # straddle in the only sense available to a move that does not roll.
    OwedCase(
        label="fixed-damage-seismictoss",
        covers="fixed-damage",
        exercises="no roll fan and no crit entry; a rolled model would straddle",
        move="seismictoss", accuracy=1.0, hits=1, rolls=False,
        hp=105, maxhp=244, status="burn",
    ),
    # (3) MULTI-HIT. Bonemerang's per-hit max is 61 against 120 HP, so a hit-count-
    # blind partition sees 61 < 120 and never fires; the two hits TOGETHER reach 122
    # and the total fan [103..122] straddles both the hit-KO threshold (case-a) and
    # the sandstorm residual threshold at 106. Three populations at once -- 2 rolls
    # kill on the hit, 12 die to sand, 2 survive -- so an arm-assignment error at any
    # of the three boundaries moves faint mass. The defender is Normal/typeless
    # rather than Normal/Flying because Bonemerang is Ground and Flying is immune to
    # it; typeless keeps the sand tick, which Ground/Rock/Steel would suppress.
    OwedCase(
        label="multihit-bonemerang-sand",
        covers="multi-hit",
        exercises="per-hit max below hp, total fan straddles hp and the residual",
        move="bonemerang", accuracy=0.9, hits=2, rolls=True,
        hp=120, maxhp=244, weather="sand",
        defender_types=("normal", "typeless"),
    ),
    # (4) MIRROR STEP 10.5 -- LEECH SEED, at the non-crit unbounded-ceiling site.
    # This is the family PR #1185's patch 74 operates on and the one the mass gate
    # most conspicuously did not reach: the drain is 244/8 = 30, the threshold 111,
    # and ten of the sixteen non-crit rolls die to it. Scope, stated because the
    # obvious reading is wrong: the KO-MASS row here covers the mirror STEP, not the
    # per-roll split. Splitting a lethal band into one arm per roll moves no mass
    # between faint and alive, so no functional built on total KO mass can see patch
    # 74 at all -- measured, not assumed (``reports/c152`` M7). What sees it is
    # ``test_the_leechseed_bands_are_split_per_roll`` below, on the SHAPE.
    OwedCase(
        label="leechseed-mirror",
        covers="mirror-step-leech-seed",
        exercises="Leech Seed drain is the threshold on the non-crit fan",
        move="rockslide", accuracy=0.9, hits=1, rolls=True,
        hp=140, maxhp=244, volatiles=("leechseed",),
    ),
    # (5) MIRROR STEP 10.5 at the OTHER site patch 74 touches -- the crit fan that
    # cannot kill. Same geometry as (1) with the sandstorm replaced by a Leech Seed
    # drain of 255/8 = 31, so the threshold is 219 and eleven of the sixteen crit
    # rolls die to it. Both of the patch's two call sites are reached by the matrix,
    # and reached SEPARATELY: (4) and (5) sit on opposite sides of the crit split, so
    # a patch that regressed at one site only still reddens.
    OwedCase(
        label="leechseed-crit-fan",
        covers="mirror-step-leech-seed",
        exercises="Leech Seed drain is the threshold inside a non-lethal crit fan",
        move="rockslide", accuracy=0.9, hits=1, rolls=True,
        hp=250, maxhp=255, volatiles=("leechseed",),
    ),
    # (6) MIRROR STEP 10.9 -- the partial trap. maxhp/16 = 15, threshold 116, six of
    # the sixteen non-crit rolls die to it. Six rather than ten deliberately: (4) and
    # (6) would otherwise share an expected mass and a single wrong number could
    # satisfy both.
    OwedCase(
        label="partial-trap-mirror",
        covers="mirror-step-partial-trap",
        exercises="partial-trap tick is the threshold on the non-crit fan",
        move="rockslide", accuracy=0.9, hits=1, rolls=True,
        hp=130, maxhp=244, volatiles=("partiallytrapped",),
    ),
    # (7) MIRROR STEP 7 -- a resolving Wish, and the one fixture here whose KO-mass
    # row is a CONTROL rather than the assertion. Wish heals min(maxhp - hp, maxhp/2)
    # BEFORE every damage tick, and maxhp/2 strictly exceeds the sum of every tick
    # the mirror models (16ths and 8ths totalling at most 3*maxhp/8), so a resolving
    # Wish makes residual death impossible -- the whole faint mass is the crit fan's
    # hit KOs, 0.9 * 1/16 = 5.625 %.
    #
    # That row cannot fail on this axis and it is not pretending to: deleting the
    # order-7 heal from the mirror leaves BOTH the engine's KO mass and this
    # reconstruction at 5.625 %, because the real end-of-turn phase still applies the
    # Wish and refuses the KO the mispriced mirror expected (``reports/c152`` M6).
    # What that mutation DOES move is the branch shape, from one collapsed arm to a
    # partitioned pair, so mirror step 7's discriminating assertion is
    # ``test_a_resolving_wish_leaves_the_fan_collapsed`` below. hp is 140 rather than
    # a lower value for exactly this reason: at 130 the mispriced threshold falls
    # BELOW the fan floor, the fan saturates, the shape stays collapsed and the
    # mutation is invisible to every assertion in this file.
    OwedCase(
        label="wish-mirror",
        covers="mirror-step-wish",
        exercises="a resolving Wish outheals every tick; no roll can die to residual",
        move="rockslide", accuracy=0.9, hits=1, rolls=True,
        hp=140, maxhp=244, status="burn", wish=(1, 0),
    ),
)


def _owed_state(case: OwedCase, attacker_move: str | None = None) -> pe.State:
    """The fixture's state. Same attacker profile as ``_state``; see OWED_CASES."""

    attacker = pe.Pokemon(
        id="gligar", level=81,
        types=("ground", "flying"), base_types=("ground", "flying"),
        hp=205, maxhp=205, ability="none", item="none",
        attack=170, defense=160, special_attack=120,
        special_defense=130, speed=250,
        moves=[pe.Move(id=attacker_move or case.move, pp=16),
               pe.Move(id="splash", pp=16)],
    )
    defender = pe.Pokemon(
        id="fearow", level=81,
        types=case.defender_types, base_types=case.defender_types,
        hp=case.hp, maxhp=case.maxhp, ability="none", item=case.item,
        attack=170, defense=145, special_attack=110,
        special_defense=125, speed=100, status=case.status,
        moves=[pe.Move(id="splash", pp=16)],
    )
    kw = {}
    if case.toxic_count is not None:
        kw["side_conditions"] = pe.SideConditions(toxic_count=case.toxic_count)
    return pe.State(
        side_one=pe.Side(active_index="0", pokemon=[attacker] + [_dummy()] * 5),
        side_two=pe.Side(
            active_index="0", pokemon=[defender] + [_dummy()] * 5,
            volatile_statuses=set(case.volatiles), wish=case.wish, **kw,
        ),
        weather=case.weather, terrain="none", trick_room=False,
    )


def _owed_tick(case: OwedCase) -> int:
    """Step (2) of the recipe, for an owed case: the phase's own quiet-turn NET."""

    quiet = _owed_state(case, "splash")
    return _net_hp_lost_by_defender(
        pe.generate_instructions(quiet, "splash", "splash")[0]
    )


def _owed_fans(case: OwedCase) -> tuple[list[int], list[int]]:
    """``(non-crit fan, crit fan)`` as TOTAL damage per roll, ascending.

    ``calculate_damage`` returns ``[non_crit_max, crit_max]`` for a rolling move and
    a ONE-element list for a ``damageCallback`` move; the length is checked against
    the fixture's own ``rolls`` descriptor rather than sniffed, so a move that grows
    or loses a crit entry fails by name instead of shifting an index.
    """

    state = _owed_state(case)
    damages = pe.calculate_damage(state, case.move, "splash", False)[0]
    if len(damages) != (2 if case.rolls else 1):
        raise AssertionError(
            f"{case.label}: calculate_damage returned {damages}; a rolls={case.rolls} "
            f"move must return {2 if case.rolls else 1} maxima. The fixture's own "
            "description of the move no longer matches the engine."
        )
    max_regular = damages[0] * case.hits
    max_crit = (damages[1] if case.rolls else damages[0]) * case.hits
    # A move that does not roll deals its maximum on all sixteen; enumerating
    # ``r in 85..=100`` for it is precisely the defect the fixture exists to catch,
    # so it must not appear on this side of the comparison.
    rolls = range(85, 101) if case.rolls else (100,) * 16
    return (
        [max_regular * r // 100 for r in rolls],
        [max_crit * r // 100 for r in rolls],
    )


def measure_owed_shapes() -> dict[str, list[int]]:
    """``measure_shapes``'s functional over ``OWED_CASES``: distinct arm damages.

    Damage VALUES, never branch counts, for the reason ``measure_shapes`` gives.
    Only the branch's FIRST ``Damage SideTwo`` is taken, so a multi-hit arm reports
    its PER-HIT value -- which is what the engine actually pushed.
    """

    shapes = {}
    for case in OWED_CASES:
        tick = _owed_tick(case)
        values = set()
        for branch in pe.generate_instructions(
            _owed_state(case), case.move, "splash"
        ):
            # Same miss test ``measure_shapes`` uses: a miss loses exactly the tick.
            if _net_hp_lost_by_defender(branch) == tick:
                continue
            for instruction in branch.instruction_list:
                text = str(instruction)
                if text.startswith("Damage SideTwo"):
                    values.add(int(text.split(": ")[1]))
                    break
        shapes[case.label] = sorted(v for v in values if v != case.hp)
    return shapes


def collapsed_shapes() -> dict[str, list[int]]:
    """``measure_shapes`` from a child process pinned to the collapsed cascade.

    A child, because ``POKEZERO_ENUMERATE_ROLLS`` is latched once per process by a
    ``OnceLock``: this parent may already have called into the engine, and on an
    enumerated parent an in-process measurement would silently report enumerated
    shapes to a control whose whole job is to find a collapsed one.

    Setting ``"0"`` rather than unsetting is deliberate and its scope is limited: it
    makes this gate independent of the ambient environment, and it would catch an
    engine that ignored the flag, but it cannot see the patch's default flip. The
    unset-environment case is asserted in ``tests/test_roll_enumeration_scope.py``.
    """
    return _collapsed_shape_probe("--emit-shapes")


def collapsed_owed_shapes() -> dict[str, list[int]]:
    """``measure_owed_shapes`` from the same collapsed child, for the same reason.

    The owed matrix's two shape pins are about the COLLAPSED cascade's arm set --
    the per-roll Leech Seed split and the Wish control. Under enumeration every fan
    is already one arm per roll, so both would report an unrelated shape and the
    pins would be measuring the wrong engine.
    """

    return _collapsed_shape_probe("--emit-owed-shapes")


def _collapsed_shape_probe(flag: str) -> dict[str, list[int]]:
    environment = subproc_env()
    environment[ENUMERATE_ROLLS_ENV] = "0"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), flag],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"collapsed shape probe ({flag}) exited "
            f"{result.returncode}\n{result.stdout}\n{result.stderr}"
        )
    return json.loads(result.stdout)


class BranchMassReconstruction(unittest.TestCase):
    """Each case: (label, hp, status, item, weather, toxic_count)."""

    CASES = (
        ("noncrit-straddles-toxic",   123, "toxic",  "none",      "none", 0),
        ("noncrit-straddles-sand",    130, "none",   "none",      "sand", None),
        ("noncrit-straddles-burn",    123, "burn",   "leftovers", "none", None),
        ("saturated-toxic-count-1",   123, "toxic",  "none",      "none", 1),
        ("straddle-poison",           140, "poison", "none",      "none", None),  # 10/16 dies -- NOT saturated
        # Reaches the C27 crit-KILL split: min_crit 207 < hp <= max_crit 244, and no
        # residual, so the crit-survive arm stays OUT of the KO set and the split's
        # proportions become load-bearing. Replaces a "crit-fan-only" fixture that
        # used hp=280 against MAXHP=244 -- an unreachable battle state, which is
        # precisely why its threshold landed above max_crit and the crit fan never
        # split. With C27's crit-kill split AND #1062's crit-fan split both corrupted
        # to crit_rate*0.5, the previous matrix passed entirely.
        ("crit-kill-straddle",        230, "none",   "none",      "none", None),
        ("case-a-three-way",          120, "burn",   "leftovers", "none", None),
        ("no-residual-at-all",        160, "none",   "none",      "none", None),
        ("inert-item-salac",          140, "burn",   "salacberry", "none", None),
    )

    def _reconstruct_and_measure(self, hp, status, item, weather, toxic_count):
        # (2) the phase reports its own tick, with no attack in play.
        quiet = _state(hp, status, item, weather, toxic_count, "splash")
        tick = _net_hp_lost_by_defender(
            pe.generate_instructions(quiet, "splash", "splash")[0]
        )
        self.assertLess(tick, hp, "invalid fixture: the residual alone would KO")
        self.assertLessEqual(
            hp, MAXHP,
            "invalid fixture: hp exceeds maxhp, an unreachable battle state. The "
            "constructor accepts it silently and it shifts every threshold, which is "
            "how a fixture came to assert 0.0 == 0.0 while reading PASS.",
        )

        state = _state(hp, status, item, weather, toxic_count, "rockslide")
        max_regular = pe.calculate_damage(state, "rockslide", "splash", False)[0][0]
        max_crit = pe.calculate_damage(state, "rockslide", "splash", True)[0][1]

        # (1) and (3): exact integer enumeration, no engine helper.
        rolls = range(85, 101)
        n_regular = sum(1 for r in rolls if hp - (max_regular * r // 100) - tick <= 0)
        n_crit = sum(
            1 for r in rolls
            if hp - (max_crit * r // 100) <= 0
            or hp - (max_crit * r // 100) - tick <= 0
        )
        expected = ACCURACY * (
            (1.0 - CRIT_RATE) * n_regular / 16.0 + CRIT_RATE * n_crit / 16.0
        ) * 100.0

        branches = pe.generate_instructions(state, "rockslide", "splash")
        actual = sum(
            b.percentage for b in branches if _net_hp_lost_by_defender(b) >= hp
        )
        return expected, actual, branches, max_regular, tick

    def test_ko_mass_matches_independent_reconstruction(self):
        for label, hp, status, item, weather, tc in self.CASES:
            with self.subTest(case=label):
                expected, actual, _, mx, tick = self._reconstruct_and_measure(
                    hp, status, item, weather, tc
                )
                self.assertAlmostEqual(
                    actual, expected, delta=0.001,
                    msg=(f"{label}: hp={hp} tick={tick} max={mx} — engine KO mass "
                         f"{actual:.4f}% vs reconstruction {expected:.4f}%. The sweep "
                         f"cannot see this class of disagreement."),
                )

    def test_the_fixture_matrix_is_the_expected_size(self):
        """The count guards in CI count test METHODS, not fixtures. Six of the nine
        CASES could be deleted with all four tests green, because the assertions run
        under subTest inside two methods. Review of #1083 found the residual one level
        down; this closes it."""
        self.assertEqual(
            len(self.CASES), 9,
            "the fixture matrix changed size; CI's test-count guards cannot see this",
        )

    def test_masses_sum_to_one(self):
        for label, hp, status, item, weather, tc in self.CASES:
            with self.subTest(case=label):
                state = _state(hp, status, item, weather, tc, "rockslide")
                total = sum(
                    b.percentage for b in pe.generate_instructions(
                        state, "rockslide", "splash"
                    )
                )
                self.assertAlmostEqual(total, 100.0, delta=0.001, msg=label)

    def test_matrix_is_not_vacuous(self):
        """A fixture that straddles nothing asserts nothing and still reads PASS.

        Six such near-misses shipped in the previous era. This asserts the matrix
        as a whole exercises the arm structures it claims to, using distinct move
        damages rather than branch counts (flinch doubles every arm).

        Measured in a child process pinned to the COLLAPSED cascade -- see the module
        docstring. The three assertions below are byte-for-byte the ones that shipped;
        only where the shapes come from has changed.
        """
        shapes = collapsed_shapes()

        multi = [k for k, v in shapes.items() if len(v) >= 2]
        single = [k for k, v in shapes.items() if len(v) == 1]
        self.assertTrue(
            multi, f"no fixture partitions a fan; the matrix proves nothing: {shapes}"
        )
        self.assertTrue(
            single, f"no fixture leaves a fan collapsed; no negative control: {shapes}"
        )
        self.assertGreaterEqual(
            len(shapes["case-a-three-way"]), 2,
            f"case-a-three-way must show a partitioned fan, got {shapes}",
        )

    def test_the_negative_control_names_its_fixture(self):
        """Which fixture is the collapsed control, and what it collapses TO.

        ``single`` being non-empty is satisfied by any one of nine fixtures, so the
        control could migrate between fixtures across an engine change and nobody
        would see it move. ``saturated-toxic-count-1`` is the designed control -- the
        residual already saturates, so no threshold straddles its fan and the
        cascade emits one representative roll. Pinning the VALUE as well as the
        count makes a representative that drifts to a different roll fail here by
        name, which a shape-length check cannot do.
        """
        shapes = collapsed_shapes()
        self.assertEqual(
            shapes["saturated-toxic-count-1"], [112],
            "the designed collapsed control changed shape; if the cascade now "
            "partitions this fan the matrix has lost its negative control",
        )

    def test_named_constants_are_pinned_by_a_named_arm(self):
        """ACCURACY and CRIT_RATE are literals, deliberately -- reading them from the
        engine would destroy the only genuine independence left. But on a fixture whose
        whole fan dies, `expected` collapses to ACCURACY*100 and CRIT_RATE cancels, so
        the constants go unconstrained. These two checks make a constant change fail by
        name rather than diffusely."""
        state = _state(160, "none", "none", "none", None, "rockslide")
        branches = pe.generate_instructions(state, "rockslide", "splash")
        miss = [b for b in branches if _net_hp_lost_by_defender(b) == 0]
        self.assertAlmostEqual(
            sum(b.percentage for b in miss), 100.0 * (1.0 - ACCURACY), delta=0.001,
            msg="miss-arm mass must equal 100*(1 - ACCURACY); ACCURACY may have drifted",
        )
        kills = [b for b in branches if _net_hp_lost_by_defender(b) >= 160]
        self.assertAlmostEqual(
            sum(b.percentage for b in kills), 100.0 * ACCURACY * CRIT_RATE, delta=0.001,
            msg="with nothing pending, only a crit kills: mass must be "
                "100*ACCURACY*CRIT_RATE; a constant may have drifted",
        )

    # -- C119 obligation 1 / c137 section 4: the four owed families. ------------

    def _reconstruct_owed(self, case: OwedCase):
        tick = _owed_tick(case)
        self.assertLess(
            tick, case.hp, f"{case.label}: the residual alone would KO",
        )
        self.assertLessEqual(
            case.hp, case.maxhp,
            f"{case.label}: hp exceeds maxhp, an unreachable battle state",
        )
        regular_fan, crit_fan = _owed_fans(case)

        # The non-crit arm keeps ``CASES``'s expression, ``hp - d - tick <= 0``,
        # rather than gaining an ``or hp - d <= 0`` guard. With ``tick >= 0`` the
        # two are the same set; with ``tick < 0`` -- a NET HEAL, which only the Wish
        # fixture has -- they are not, and the guard would be the only untested
        # branch in this file. ``test_the_owed_matrix_is_not_vacuous`` instead
        # requires any net-heal fixture to keep its whole non-crit fan below ``hp``,
        # which makes the expression exact rather than merely guarded.
        n_regular = sum(1 for d in regular_fan if case.hp - d - tick <= 0)
        n_crit = sum(
            1 for d in crit_fan
            if case.hp - d <= 0 or case.hp - d - tick <= 0
        )
        expected = case.accuracy * (
            (1.0 - CRIT_RATE) * n_regular / 16.0 + CRIT_RATE * n_crit / 16.0
        ) * 100.0

        branches = pe.generate_instructions(
            _owed_state(case), case.move, "splash"
        )
        actual = sum(
            b.percentage for b in branches
            if _net_hp_lost_by_defender(b) >= case.hp
        )
        return expected, actual, tick, regular_fan, crit_fan

    def test_owed_family_ko_mass_matches_independent_reconstruction(self):
        """The gate, over the owed matrix. Same functional as ``CASES``'s."""
        for case in OWED_CASES:
            with self.subTest(case=case.label):
                expected, actual, tick, fan, crit_fan = self._reconstruct_owed(case)
                self.assertAlmostEqual(
                    actual, expected, delta=0.001,
                    msg=(f"{case.label} ({case.covers}): hp={case.hp} tick={tick} "
                         f"fan=[{fan[0]}..{fan[-1]}] crit=[{crit_fan[0]}..{crit_fan[-1]}] "
                         f"— engine KO mass {actual:.4f}% vs reconstruction "
                         f"{expected:.4f}%. The sweep cannot see this class of "
                         f"disagreement."),
                )

    def test_owed_family_masses_sum_to_one(self):
        """Free, weaker, and the only thing here that catches an in-place
        ``update_percentage`` leak rather than a misplaced roll."""
        for case in OWED_CASES:
            with self.subTest(case=case.label):
                total = sum(
                    b.percentage for b in pe.generate_instructions(
                        _owed_state(case), case.move, "splash"
                    )
                )
                self.assertAlmostEqual(total, 100.0, delta=0.001, msg=case.label)

    def test_the_owed_matrix_is_the_expected_size(self):
        """The count guards in CI count test METHODS. Six of the seven owed
        fixtures could be deleted with every method above still green, because they
        all run under ``subTest``. This is the guard one level down, and the same
        one ``test_the_fixture_matrix_is_the_expected_size`` provides for ``CASES``.
        """
        self.assertEqual(
            len(OWED_CASES), 7,
            "the owed fixture matrix changed size; CI's test-count guards cannot "
            "see this",
        )
        self.assertEqual(
            sorted(c.label for c in OWED_CASES),
            [
                "crit-fan-residual-sand",
                "fixed-damage-seismictoss",
                "leechseed-crit-fan",
                "leechseed-mirror",
                "multihit-bonemerang-sand",
                "partial-trap-mirror",
                "wish-mirror",
            ],
        )

    def test_every_owed_family_has_a_fixture_or_a_reachability_verdict(self):
        """The discharge itself, asserted rather than claimed in a comment.

        c137 section 4 records four owed families and this file's docstring records
        the same four. Every one of them must be reached by a fixture here, or
        carry a pool-reachability verdict in ``POOL_UNREACHABLE`` -- which is a
        MEASUREMENT, not a waiver, in the convention of
        ``reports/c138_known_gaps_ledger.md`` section 1.1.

        Written as a total mapping in both directions on purpose. "Every family is
        covered" alone would stay green if a fixture drifted onto a family name
        that is not owed, and "every fixture names an owed family" alone would stay
        green if a family lost its last fixture.

        WHAT THIS DOES NOT CATCH, measured by review of #1198 rather than reasoned
        about. It catches a family losing its LAST fixture -- relabelling BOTH
        Leech Seed fixtures reddens it. It does NOT catch a MISLABEL that leaves
        both families populated: repointing ``leechseed-crit-fan``'s ``covers`` at
        ``mirror-step-partial-trap`` leaves every family accounted for and the
        whole module reads 14/14 OK, while Leech Seed silently drops from two
        fixtures to one. What holds that line instead is
        ``test_the_owed_matrix_is_not_vacuous``, which asserts each fixture's SHAPE
        structurally -- a mislabelled fixture keeps its shape, so the mislabel is a
        documentation defect rather than a coverage one, but this method is not the
        thing that would object and its docstring should not imply otherwise.
        """
        covered = {case.covers for case in OWED_CASES}
        accounted = covered | set(POOL_UNREACHABLE)
        missing = sorted(set(OWED_FAMILIES) - accounted)
        self.assertEqual(
            missing, [],
            f"no fixture and no reachability verdict for {missing}; c137 section 4's "
            f"obligation is not discharged for them. Reached: {sorted(covered)}",
        )
        stray = sorted(accounted - set(OWED_FAMILIES))
        self.assertEqual(
            stray, [],
            f"{stray} is named by a fixture or a verdict but is not an owed family",
        )
        overlap = sorted(covered & set(POOL_UNREACHABLE))
        self.assertEqual(
            overlap, [],
            f"{overlap} is BOTH shipped as a fixture and recorded unreachable; one "
            f"of the two is wrong and the fixture is the more likely",
        )

    def test_the_rain_dish_waiver_is_backed_by_the_committed_census(self):
        """The one family discharged WITHOUT a fixture must show its measurement.

        Rain Dish is waived, not covered, so the waiver is the only thing standing
        between this branch and an undischarged obligation. Review of #1198 found
        the figures behind it living in prose while c137's bullet called them
        "machine-checked" -- nothing re-derived 71 abilities / 0 Rain Dish. This is
        that assertion.

        SCOPE, because the obvious reading is again too strong. This compares
        ``POOL_UNREACHABLE``'s figures against a COMMITTED census
        (``scripts/c152_pool_reachability_census.py``, regenerated out of process).
        It therefore catches the waiver's prose drifting from its own measurement,
        and it makes any change to the numbers a deliberate, reviewable edit. It
        does NOT re-derive them against a live pool: CI builds no Showdown checkout
        and this module's workflow step forbids skips, so a live derivation cannot
        run here at all. A Showdown bump that put Rain Dish on a gen3 set would
        leave artifact, gate and waiver green and wrong -- which is why the
        ``showdown_commit`` the census was taken at is one of the pinned figures.
        """
        census = json.loads(POOL_CENSUS_PATH.read_text(encoding="utf-8"))
        waiver = POOL_UNREACHABLE["mirror-step-rain-dish"]
        self.assertEqual(waiver["verdict"], "UNREACHABLE")

        counts = waiver["counts"]
        for key in ("showdown_commit", "species", "sets", "distinct_abilities"):
            self.assertEqual(
                counts[key], census[key],
                f"the Rain Dish waiver says {key}={counts[key]!r} and the committed "
                f"census says {census[key]!r}. Regenerating the census is NOT the "
                f"fix unless the pool really moved; the waiver is what changes.",
            )
        for ability in ("Rain Dish", "Dry Skin"):
            self.assertEqual(
                counts[ability], census["named_abilities"][ability],
                f"the waiver and the census disagree about {ability}",
            )
            self.assertEqual(
                census["named_abilities"][ability], 0,
                f"{ability} is now in the gen3 randbats pool, so mirror step 10.3 "
                f"is REACHABLE and c137 §4's obligation is no longer discharged for "
                f"it by a verdict. It needs a fixture.",
            )
        # The two halves of "unreachable in the POOL, not in gen3". If the step
        # stopped existing in gen3 the waiver would still be true but its stated
        # reason would be wrong, and a reader would carry the wrong model forward.
        self.assertTrue(census["raindish_exists_in_gen3"])
        self.assertEqual(census["raindish_num"], 44)
        # Trace is the only pool mechanism that could import an absent ability, and
        # the waiver's note turns on it being present-but-powerless here.
        self.assertGreater(
            census["named_abilities"]["Trace"], 0,
            "Trace has left the pool; the waiver's cross-side note now describes a "
            "mechanism that is not there, even though its conclusion still holds",
        )

    def test_the_owed_matrix_is_not_vacuous(self):
        """Each owed fixture must actually reach the shape it is named for.

        A fixture that straddles nothing asserts nothing and still reads PASS; six
        such shipped in the previous era. These checks are structural -- computed
        from the fan and from the phase's own quiet-turn tick -- so they stay true
        if the partition code changes and false if the fixture drifts.
        """
        shapes = {}
        for case in OWED_CASES:
            tick = _owed_tick(case)
            regular_fan, crit_fan = _owed_fans(case)
            shapes[case.label] = {
                "tick": tick,
                # Smallest damage that is residual-lethal, i.e. the threshold this
                # gate's one residual model implies. ``None`` when the phase heals
                # net, which is unreachable by any damage.
                "kill_floor": (case.hp - tick) if tick > 0 else None,
                "regular_fan": (regular_fan[0], regular_fan[-1]),
                "crit_fan": (crit_fan[0], crit_fan[-1]),
                "hp": case.hp,
            }

        # Every fixture whose tick is a NET HEAL must keep its whole non-crit fan
        # below hp, which is what makes ``_reconstruct_owed``'s non-crit expression
        # exact without an extra guard. See the comment there.
        for case in OWED_CASES:
            s = shapes[case.label]
            if s["tick"] < 0:
                self.assertLess(
                    s["regular_fan"][1], s["hp"],
                    f"{case.label}: a net-heal tick with a hit-lethal non-crit fan "
                    f"would make the reconstruction understate the KO mass: {s}",
                )

        # (1) and (5): the crit fan must fail to kill on the hit, or these reach the
        # crit-STRADDLE site instead and the crit-FAN site stays unguarded. The
        # non-crit fan must not reach the threshold, or the fixture does not isolate
        # the crit site.
        for label in ("crit-fan-residual-sand", "leechseed-crit-fan"):
            s = shapes[label]
            self.assertLess(
                s["crit_fan"][1], s["hp"],
                f"{label}: the crit fan can kill on the hit, so this is the "
                f"crit-straddle site and the crit-FAN site is unguarded: {s}",
            )
            self.assertIsNotNone(s["kill_floor"], f"{label}: no residual at all: {s}")
            self.assertTrue(
                s["crit_fan"][0] < s["kill_floor"] <= s["crit_fan"][1],
                f"{label}: the residual threshold is not inside the crit fan: {s}",
            )
            self.assertGreater(
                s["kill_floor"], s["regular_fan"][1],
                f"{label}: the NON-crit fan reaches the threshold too, so this does "
                f"not isolate the crit site: {s}",
            )

        # (2) fixed damage: one point, no crit entry, and -- the load-bearing part --
        # a rolled model of the SAME move would straddle the threshold. Without that
        # last check the fixture would read PASS on any hp at all.
        fixed = next(c for c in OWED_CASES if c.covers == "fixed-damage")
        s = shapes[fixed.label]
        self.assertFalse(fixed.rolls, "the fixed-damage fixture must not roll")
        self.assertEqual(
            s["regular_fan"], s["crit_fan"],
            f"{fixed.label}: a damageCallback move cannot crit: {s}",
        )
        self.assertEqual(
            s["regular_fan"][0], s["regular_fan"][1],
            f"{fixed.label}: the fan is not a single point: {s}",
        )
        self.assertIsNotNone(s["kill_floor"])
        self.assertTrue(
            (s["regular_fan"][0] * 85 // 100) < s["kill_floor"] <= s["regular_fan"][0],
            f"{fixed.label}: a roll-scaled Seismic Toss would NOT straddle this "
            f"threshold, so the fixture asserts nothing about fixed damage: {s}",
        )

        # (3) multi-hit: a hit-count-blind partition must miss it, and the total fan
        # must straddle both the hit-KO threshold and the residual threshold.
        multi = next(c for c in OWED_CASES if c.covers == "multi-hit")
        s = shapes[multi.label]
        self.assertGreater(multi.hits, 1)
        self.assertLess(
            s["regular_fan"][1] // multi.hits, s["hp"],
            f"{multi.label}: the PER-HIT max already reaches hp, so a hit-count-"
            f"blind partition would fire anyway and the scaling is unguarded: {s}",
        )
        self.assertTrue(
            s["regular_fan"][0] < s["hp"] <= s["regular_fan"][1],
            f"{multi.label}: the total fan does not straddle hp, so this is not the "
            f"case-a site: {s}",
        )
        self.assertIsNotNone(s["kill_floor"])
        self.assertTrue(
            s["regular_fan"][0] < s["kill_floor"] < s["hp"],
            f"{multi.label}: no residual band exists strictly under the KO "
            f"threshold, so hp is not acting as the band ceiling: {s}",
        )

        # (4) and (6): the non-crit mirror-step fixtures. The tick must be the step's
        # own magnitude -- Leech Seed maxhp/8, partial trap maxhp/16 -- so a fixture
        # that quietly acquired a second residual source stops claiming to isolate
        # one step, and the threshold must sit strictly inside the non-crit fan.
        for label, magnitude in (
            ("leechseed-mirror", lambda c: int(c.maxhp * 0.125)),
            ("partial-trap-mirror", lambda c: c.maxhp // 16),
        ):
            case = next(c for c in OWED_CASES if c.label == label)
            s = shapes[label]
            self.assertEqual(
                s["tick"], magnitude(case),
                f"{label}: the quiet-turn tick is not this mirror step's own "
                f"magnitude, so the fixture no longer isolates it: {s}",
            )
            self.assertTrue(
                s["regular_fan"][0] < s["kill_floor"] <= s["regular_fan"][1],
                f"{label}: the threshold is not inside the non-crit fan, so no roll "
                f"straddles it: {s}",
            )

        # (7) Wish: the step must be doing the thing the fixture claims -- healing
        # net, and by enough that no roll of either fan can die to the residual.
        wish = next(c for c in OWED_CASES if c.covers == "mirror-step-wish")
        s = shapes[wish.label]
        self.assertLess(
            s["tick"], 0,
            f"{wish.label}: the Wish is not resolving, so the fixture is an "
            f"ordinary burn fixture: {s}",
        )
        self.assertIsNone(s["kill_floor"])
        # And the mirror WOULD have found a threshold inside the fan without it --
        # otherwise the fan saturates, the shape stays collapsed either way, and the
        # pin below cannot fail. Burn alone ticks maxhp/8.
        without_wish = wish.hp - wish.maxhp // 8
        self.assertTrue(
            s["regular_fan"][0] < without_wish <= s["regular_fan"][1],
            f"{wish.label}: without the Wish heal the threshold would fall outside "
            f"the fan, so removing the heal changes nothing and the collapsed-shape "
            f"pin below is inert: {s} without_wish={without_wish}",
        )

    def test_the_leechseed_bands_are_split_per_roll(self):
        """PR #1185 / patch 74, at BOTH of the two sites it edits.

        The newest engine patch splits a residual-kill band into one arm per roll
        where the killing residual is Leech Seed. No functional built on total KO
        mass can see it -- every arm of a lethal band is lethal before and after, so
        the split moves no mass between faint and alive, measured rather than
        assumed (``reports/c152`` M7 leaves every KO-mass row in this file green).
        The SHAPE is what moves, so the shape is what is pinned.

        Values, not counts, and the exact list rather than a length: an arm that
        drifts to a damage no legal roll deals is precisely the failure the patch's
        integer fan exists to prevent, and a length check cannot see it. The lists
        below are the sixteen-roll integer fan restricted to each band, plus the
        band's survive representative.
        """
        shapes = collapsed_owed_shapes()
        self.assertEqual(
            shapes["leechseed-mirror"],
            [106, 111, 112, 113, 114, 115, 117, 118, 119, 120, 122],
            "the non-crit Leech Seed band is no longer split per roll (patch 74's "
            "first site). Collapsed, this reads [106, 110]: one arm at the "
            "threshold and the survive representative.",
        )
        self.assertEqual(
            shapes["leechseed-crit-fan"],
            [112, 211, 219, 222, 224, 226, 229, 231, 234, 236, 239, 241, 244],
            "the crit Leech Seed band is no longer split per roll (patch 74's "
            "second site). Collapsed, this reads [112, 211, 219].",
        )

    def test_a_resolving_wish_leaves_the_fan_collapsed(self):
        """Mirror step 7's discriminating assertion; see the ``wish-mirror`` comment.

        A resolving Wish heals more than every tick the mirror models combined, so
        no roll can die to the residual and the mirror must find no threshold at
        all. Deleting the order-7 heal gives it one, at 110, strictly inside the fan
        -- the fan then partitions into [106, 110] while the KO mass, which the real
        end-of-turn phase still decides, does not move by a thousandth of a point.
        """
        shapes = collapsed_owed_shapes()
        self.assertEqual(
            shapes["wish-mirror"], [112],
            "the fan under a resolving Wish is partitioned; the mirror has found a "
            "residual threshold that a Wish makes unreachable",
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--emit-shapes"]:
        # Child entry point for ``collapsed_shapes``. Not a test: it reports the
        # shapes of whatever roll path this process was started in.
        print(json.dumps(measure_shapes()))
        raise SystemExit(0)
    if sys.argv[1:2] == ["--emit-owed-shapes"]:
        # Child entry point for ``collapsed_owed_shapes``; same contract.
        print(json.dumps(measure_owed_shapes()))
        raise SystemExit(0)
    unittest.main(verbosity=2)
