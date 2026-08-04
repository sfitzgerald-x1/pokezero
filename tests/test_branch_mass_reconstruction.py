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

Still uncovered: the crit-FAN residual split (needs ``max_crit < hp``, a second
attacker profile rather than another hp value), ``fixed_damage``, multi-hit moves, the
Wish / Rain Dish / Leech Seed / partial-trap mirror steps, and the bail set. The bail
set is unreachable BY THIS DESIGN — a scalar quiet-turn tick cannot represent Sitrus's
non-monotone threshold heal — so covering it needs a different reconstruction, not
another fixture. These are an obligation on the Phase 2 decision record: if the
partition stack is RETAINED for any consumer (plan outcomes (b) or (c)), they get
fixtures before that decision is recorded as closed.

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
"""

from __future__ import annotations

import unittest

import poke_engine as pe

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
        """
        shapes = {}
        for label, hp, status, item, weather, tc in self.CASES:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
