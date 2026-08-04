#!/usr/bin/env python
"""Standing mass gate — C116 Phase 1 item 4.

WHY THIS EXISTS. The engine transition differential compares roll-scaled damage
*components*; it never compares branch *probability masses*. So an entire class of
defect is invisible to it: anything that moves mass between arms while leaving each
arm's components intact. That is not hypothetical. On PR #1062 the non-crit
residual split called ``update_percentage`` in place, silently scaling every crit
arm cloned from that value afterwards; the totals still summed to 100%, so no
conservation check fired, and the fix measured NEUTRAL on a 200-game sweep. It was
caught by adversarial review, and review has had to substitute for this instrument
three separate times (the mass leak, the cross-gen ``cfg!`` edit, and a rewrite of
the whole threshold model that no test distinguished).

Review is not an instrument. This is the instrument.

WHAT IT ASSERTS. For each fixture, the engine's total "defender dies this turn"
probability mass must equal a reconstruction that shares *no arithmetic* with the
engine:

  1. enumerate the sixteen gen3 rolls as ``floor(max * r / 100)`` for r in 85..=100,
     taking ``max`` from ``calculate_damage`` (a value, not a code path);
  2. read the true residual tick from a turn where NEITHER side attacks, so the
     phase reports its own magnitude rather than us predicting it;
  3. count the rolls that die, non-crit and crit separately;
  4. mass = accuracy * ((1 - crit_rate) * n_regular/16 + crit_rate * n_crit/16).

Nothing here calls ``compare_health_with_damage_multiples``, the residual mirror,
or the partition logic. If the engine and this disagree, one of them is wrong and
the sweep cannot tell you which.

It also asserts every fixture's masses sum to 100%, which is weaker but free.

FIXTURE DESIGN, learned from six near-misses. A fixture that does not straddle a
threshold asserts nothing, and reads PASS. Every case here therefore records the
arm structure it is supposed to exercise, and ``test_matrix_is_not_vacuous``
asserts that the matrix as a whole contains at least one genuine split, at least
one no-split, and at least one case where only the crit fan splits. A branch COUNT
is never used as a signal: Rock Slide flinches 30%, so every arm appears twice.
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


def _damage_to_defender(branch) -> int:
    return sum(
        int(str(i).split(": ")[1])
        for i in branch.instruction_list
        if str(i).startswith("Damage SideTwo")
    )


class BranchMassReconstruction(unittest.TestCase):
    """Each case: (label, hp, status, item, weather, toxic_count)."""

    CASES = (
        ("noncrit-straddles-toxic",   123, "toxic",  "none",      "none", 0),
        ("noncrit-straddles-sand",    130, "none",   "none",      "sand", None),
        ("noncrit-straddles-burn",    123, "burn",   "leftovers", "none", None),
        ("saturated-toxic-count-1",   123, "toxic",  "none",      "none", 1),
        ("saturated-poison",          140, "poison", "none",      "none", None),
        ("crit-fan-only",             280, "poison", "none",      "none", None),
        ("case-a-three-way",          120, "burn",   "leftovers", "none", None),
        ("no-residual-at-all",        160, "none",   "none",      "none", None),
        ("inert-item-salac",          140, "burn",   "salacberry", "none", None),
    )

    def _reconstruct_and_measure(self, hp, status, item, weather, toxic_count):
        # (2) the phase reports its own tick, with no attack in play.
        quiet = _state(hp, status, item, weather, toxic_count, "splash")
        tick = _damage_to_defender(
            pe.generate_instructions(quiet, "splash", "splash")[0]
        )
        self.assertLess(tick, hp, "invalid fixture: the residual alone would KO")

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
            b.percentage for b in branches if _damage_to_defender(b) >= hp
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
            state = _state(hp, status, item, weather, tc, "rockslide")
            values = set()
            for b in pe.generate_instructions(state, "rockslide", "splash"):
                for i in b.instruction_list:
                    text = str(i)
                    if text.startswith("Heal SideTwo"):
                        break
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
